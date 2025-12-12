# From https://github.com/vllm-project/vllm/pull/12609
from contextlib import contextmanager
from typing import Any, Dict, List, Literal, Optional, Set, Type, Union

import torch
try:
    from vllm.adapter_commons.utils import (add_adapter_worker,
                                            apply_adapters_worker,
                                            list_adapters_worker,
                                            set_active_adapters_worker)
except ImportError:
    add_adapter_worker = apply_adapters_worker = list_adapters_worker = set_active_adapters_worker = None
try:
    from vllm.adapter_commons.worker_manager import AbstractWorkerManager
except ImportError:
    # Create a dummy base class when the import fails
    # https://github.com/vllm-project/vllm/pull/25045/files
    class AbstractWorkerManager:
        def __init__(self, device):
            self.device = device

from vllm.config import LoRAConfig
from vllm.logger import init_logger
try:
    from vllm.lora.models import (LoRAModel, LoRAModelManager,
                                LRUCacheLoRAModelManager, create_lora_manager)
except ImportError:
    # Newer vLLM version moved/split lora methods
    # https://github.com/vllm-project/vllm/pull/30253
    from vllm.lora.lora_model import LoRAModel
    from vllm.lora.model_manager import LoRAModelManager, LRUCacheLoRAModelManager, create_lora_manager
from vllm.lora.peft_helper import PEFTHelper
from vllm.lora.request import LoRARequest
from vllm.lora.utils import get_adapter_absolute_path

logger = init_logger(__name__)

import inspect
import functools

@functools.lru_cache(1)
def dummy_lora_has_scaling_factor(create_dummy_lora):
    # create_dummy_lora(self, lora_id, rank, scaling_factor, embedding_modules)
    # create_dummy_lora(self, lora_id, rank, embedding_modules)
    keys = inspect.signature(create_dummy_lora).parameters.keys()
    return "scaling_factor" in keys
pass

class WorkerLoRAManager(AbstractWorkerManager):
    """WorkerLoRAManager that manages LoRA models on the worker side.

    Every request, the requested LoRAs will be loaded (unless they are already
    loaded), and every other LoRA will be unloaded."""

    _manager_cls: Type[LoRAModelManager] = LoRAModelManager

    @contextmanager
    def dummy_lora_cache(self):
        """Use this context manager to reuse the dummy lora model
        to avoid creating it repeatedly."""
        self._cached_dummy_lora = None
        yield
        self._cached_dummy_lora = False

    @property
    def is_enabled(self) -> bool:
        return True

    def create_lora_manager(
        self,
        model: torch.nn.Module,
    ) -> Any:
        lora_manager = create_lora_manager(
            model,
            max_num_seqs=self.max_num_seqs,
            max_num_batched_tokens=self.max_num_batched_tokens,
            vocab_size=self.vocab_size,
            lora_config=self.lora_config,
            device=self.device,
            lora_manager_cls=self._manager_cls,
        )
        self._adapter_manager = lora_manager
        return lora_manager.model

    def _load_adapter(self, lora_request: LoRARequest) -> LoRAModel:
        try:
            model = self._adapter_manager.model
            try:
                supported_lora_modules = model.supported_lora_modules
                packed_modules_mapping = model.packed_modules_mapping
            except:
                # vLLM 0.8.0 changed to self._adapter_manager
                supported_lora_modules = self._adapter_manager.supported_lora_modules
                packed_modules_mapping = self._adapter_manager.packed_modules_mapping
            pass
            expected_lora_modules: List[str] = []
            for module in supported_lora_modules:
                if module in packed_modules_mapping:
                    expected_lora_modules.extend(
                        packed_modules_mapping[module])
                else:
                    expected_lora_modules.append(module)

            expected_lora_modules = list(set(expected_lora_modules))

            if lora_request.lora_path:
                lora_path = get_adapter_absolute_path(lora_request.lora_path)

                peft_helper = PEFTHelper.from_local_dir(
                    lora_path, self.max_position_embeddings)
            else:
                lora_request.lora_config["vllm_max_position_embeddings"] = self.max_position_embeddings
                peft_helper = PEFTHelper.from_dict(lora_request.config)
            # Validates the LoRA configuration against requirements before
            # loading weights, throwing an exception if validation fails.
            peft_helper.validate_legal(self.lora_config)

            # For some models like Qwen2VL, we need to use hf_to_vllm_mapper
            # to ensure correct loading of lora weights.
            hf_to_vllm_mapper = None
            if (hasattr(model, "hf_to_vllm_mapper")
                    and model.hf_to_vllm_mapper is not None):
                hf_to_vllm_mapper = model.hf_to_vllm_mapper

            # Prepare common arguments
            lora_extra_vocab_size = getattr(self.lora_config, "lora_extra_vocab_size", 0)
            kwargs = {
                "lora_model_id": lora_request.lora_int_id,
                "peft_helper": peft_helper,
                "dtype": self.lora_config.lora_dtype,
                "weights_mapper": hf_to_vllm_mapper,
            }

            if getattr(lora_request, "lora_tensors", None) is not None:
                load_method = self._lora_model_cls.from_lora_tensors
                kwargs["tensors"] = lora_request.lora_tensors
                kwargs["device"] = None # Keep whatever the original device was
            else:
                load_method = self._lora_model_cls.from_local_checkpoint
                kwargs["lora_dir"] = lora_path
                kwargs["expected_lora_modules"] = expected_lora_modules
                kwargs["device"] = "cpu" # Local checkpoint is CPU

            # Check signature for backward compatibility
            sig = inspect.signature(load_method)
            if "model_vocab_size" in sig.parameters:
                kwargs["model_vocab_size"] = self.vocab_size + lora_extra_vocab_size
            else:
                kwargs["target_embedding_padding"] = self.vocab_size + lora_extra_vocab_size
                kwargs["embedding_modules"] = self.embedding_modules
                kwargs["embedding_padding_modules"] = self.embedding_padding_modules

            lora = load_method(**kwargs)

        except FileNotFoundError as e:
            # FileNotFoundError should be raised if both
            # - No adapter found to download from huggingface (or in
            #       offline mode)
            # - No local adapter files found at `lora_request.lora_path`
            # For NotFoundError
            raise ValueError(
                f"Loading lora {lora_request.lora_name} failed: No adapter "
                f"found for {lora_path}") from e
        except Exception as e:
            # For BadRequestError
            raise e

        if getattr(lora, 'extra_vocab_size', 0) > lora_extra_vocab_size:
            raise ValueError(f"LoRA added vocab size {lora.extra_vocab_size} "
                             f"is greater than lora_extra_vocab_size "
                             f"{lora_extra_vocab_size}.")

        return lora

    def add_dummy_lora(self, lora_request: LoRARequest, rank: int) -> bool:
        if lora_request.lora_int_id in self.list_adapters():
            return False
        if isinstance(self._cached_dummy_lora, LoRAModel):
            dummy_lora = self._cached_dummy_lora.clone(
                lora_request.lora_int_id)
        else:
            f = self._adapter_manager.create_dummy_lora
            if dummy_lora_has_scaling_factor(f):
                dummy_lora = f(
                    lora_id = lora_request.lora_int_id,
                    rank = rank,
                    scaling_factor = 1,
                    embedding_modules = self.embedding_modules,
                )
            else:
                dummy_lora = f(
                    lora_id = lora_request.lora_int_id,
                    rank = rank,
                    # scaling_factor = 1,
                    embedding_modules = self.embedding_modules,
                )
            if self._cached_dummy_lora is None:
                self._cached_dummy_lora = dummy_lora
        return self._adapter_manager.add_adapter(dummy_lora)

    def pin_adapter(self, adapter_id: int) -> bool:
        return self._adapter_manager.pin_adapter(adapter_id)

    def set_active_adapters(self, requests: Set[Any],
                            mapping: Optional[Any]) -> None:
        if set_active_adapters_worker:
            set_active_adapters_worker(requests, mapping, self._apply_adapters,
                                   self._adapter_manager.set_adapter_mapping)
        else:
            self._apply_adapters(requests)
            if mapping is not None:
                self._adapter_manager.set_adapter_mapping(mapping)


    def _apply_adapters(self, adapter_requests: Set[Any]) -> None:
        if apply_adapters_worker:
            apply_adapters_worker(adapter_requests, self.list_adapters,
                                self._adapter_manager.adapter_slots,
                                self.remove_adapter, self.add_adapter)
        else:
            existing_adapters = self.list_adapters()
            models_map = {
                adapter_request.adapter_id: adapter_request
                for adapter_request in adapter_requests if adapter_request
            }
            if len(models_map) > self._adapter_manager.adapter_slots:
                raise RuntimeError(
                    f"Number of requested models ({len(models_map)}) is greater "
                    "than the number of GPU model slots "
                    f"({self._adapter_manager.adapter_slots}).")
            requested_ids = set(models_map)
            for adapter_id in existing_adapters - requested_ids:
                self.remove_adapter(adapter_id)
            for adapter_id in requested_ids - existing_adapters:
                self.add_adapter(models_map[adapter_id])

    def add_adapter(self, adapter_request: Any) -> bool:
        if add_adapter_worker:
            return add_adapter_worker(adapter_request, self.list_adapters,
                                      self._load_adapter,
                                      self._adapter_manager.add_adapter,
                                      self._adapter_manager.activate_adapter)
        else:
            if adapter_request.adapter_id in self.list_adapters():
                return False
            loaded_adapter = self._load_adapter(adapter_request)
            loaded = self._adapter_manager.add_adapter(loaded_adapter)
            self._adapter_manager.activate_adapter(loaded_adapter.id)
            return loaded

    def remove_adapter(self, adapter_id: int) -> bool:
        return self._adapter_manager.remove_adapter(adapter_id)

    def remove_all_adapters(self):
        self._adapter_manager.remove_all_adapters()

    def list_adapters(self) -> Set[int]:
        if list_adapters_worker:
            return list_adapters_worker(self._adapter_manager.list_adapters)
        else:
            return set(self._adapter_manager.list_adapters())


# from vllm try to import WorkerLoRAManager
try:
    from vllm.lora.worker_manager import WorkerLoRAManager as vllm_WorkerLoRAManager
except:
    vllm_WorkerLoRAManager = None
pass

def old_init(
    self,
    max_num_seqs: int,
    max_num_batched_tokens: int,
    vocab_size: int,
    lora_config: LoRAConfig,
    device: torch.device,
    embedding_modules: Dict[str, str],
    embedding_padding_modules: List[str],
    lora_model_cls: Type[LoRAModel] = LoRAModel,
    max_position_embeddings: Optional[int] = None,
):
    self._lora_model_cls = lora_model_cls
    self.embedding_modules = embedding_modules
    self.embedding_padding_modules = embedding_padding_modules
    self._cached_dummy_lora: Union[None, Literal[False], LoRAModel] = False
    self.max_num_seqs = max_num_seqs
    self.max_num_batched_tokens = max_num_batched_tokens
    self.vocab_size = vocab_size
    self.lora_config = lora_config
    self.max_position_embeddings = max_position_embeddings
    AbstractWorkerManager.__init__(self, device)
    # Lazily initialized by create_lora_manager.
    self._adapter_manager: LoRAModelManager

from vllm.config import VllmConfig
def new_init(
    self,
    vllm_config: VllmConfig,
    device: torch.device,
    embedding_modules: dict[str, str],
    embedding_padding_modules: Optional[list[str]] = [],
    lora_model_cls: type[LoRAModel] = LoRAModel,
):

    old_init(
        self,
        max_num_seqs = vllm_config.scheduler_config.max_num_seqs,
        max_num_batched_tokens = vllm_config.scheduler_config.max_num_batched_tokens,
        vocab_size = vllm_config.model_config.get_vocab_size(),
        lora_config = vllm_config.lora_config,
        device = device,
        embedding_modules = embedding_modules,
        embedding_padding_modules = embedding_padding_modules,
        lora_model_cls = lora_model_cls,
        max_position_embeddings = vllm_config.model_config.hf_config.get_text_config().max_position_embeddings,
    )


if vllm_WorkerLoRAManager is not None:
    # get the signature and check if vllm_config is in the signature
    vllm_WorkerLoRAManager_signature = inspect.signature(vllm_WorkerLoRAManager.__init__)
    vllm_WorkerLoRAManager_signature_keys = vllm_WorkerLoRAManager_signature.parameters.keys()
    if "vllm_config" in vllm_WorkerLoRAManager_signature_keys:
        WorkerLoRAManager.__init__ = new_init
    else:
        WorkerLoRAManager.__init__ = old_init


class LRUCacheWorkerLoRAManager(WorkerLoRAManager):
    """WorkerLoRAManager that manages LoRA models on the worker side.

    Uses an LRU Cache. Every request, the requested LoRAs will be loaded
    (unless they are already loaded) and least recently used LoRAs will
    be unloaded if the cache is above capacity."""

    _manager_cls: Type[LRUCacheLoRAModelManager] = LRUCacheLoRAModelManager

    def create_lora_manager(
        self,
        model: torch.nn.Module,
    ) -> Any:
        lora_manager = create_lora_manager(
            model,
            lora_manager_cls=self._manager_cls,
            max_num_seqs=self.max_num_seqs,
            vocab_size=self.vocab_size,
            lora_config=self.lora_config,
            device=self.device,
            max_num_batched_tokens=self.max_num_batched_tokens,
        )
        self._adapter_manager = lora_manager
        return lora_manager.model

    def _apply_adapters(self, lora_requests: Set[LoRARequest]) -> None:
        loras_map = {
            lora_request.lora_int_id: lora_request
            for lora_request in lora_requests if lora_request
        }
        if len(loras_map) > self._adapter_manager.lora_slots:
            raise RuntimeError(
                f"Number of requested LoRAs ({len(loras_map)}) is greater "
                "than the number of GPU LoRA slots "
                f"({self._adapter_manager.lora_slots}).")
        for lora in loras_map.values():
            self.add_adapter(lora)

    def add_adapter(self, lora_request: LoRARequest) -> bool:
        if lora_request.lora_int_id not in self.list_adapters():
            # Load the new adapter first to ensure it is actually valid, before
            # evicting any existing adapters.
            # This may cause the # of loaded lora adapters to very temporarily
            # exceed `--max-cpu-loras`.
            lora = self._load_adapter(lora_request)

            # Loading succeeded, now check if we will exceed cache capacity and
            # evict if the oldest adapter if so
            if len(self._adapter_manager) + 1 > self._adapter_manager.capacity:
                assert isinstance(self._adapter_manager,
                                  LRUCacheLoRAModelManager)
                self._adapter_manager.remove_oldest_adapter()
            # Then add the new adapter to the cache
            loaded = self._adapter_manager.add_adapter(lora)
        else:
            # If the lora is already loaded, just touch it to
            # update its position in the caches
            loaded = self._adapter_manager.get_adapter(
                lora_request.lora_int_id) is not None
        self._adapter_manager.activate_adapter(lora_request.lora_int_id)
        return loaded
