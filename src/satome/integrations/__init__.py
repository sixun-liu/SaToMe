"""Thin runtime integrations for supported diffusion MLLM backbones."""

from .llada_o_t2i import LLaDAOT2IRuntime, enable_llada_o_t2i

__all__ = ["LLaDAOT2IRuntime", "enable_llada_o_t2i"]
