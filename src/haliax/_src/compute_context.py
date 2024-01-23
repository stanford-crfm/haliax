import contextlib
import threading
import typing
from contextlib import AbstractContextManager
from typing import Optional

import jmp
from jax.sharding import Mesh

import haliax
from haliax.partitioning import ResourceMapping


def current_mp_policy() -> jmp.Policy:
    return _mapping_holder.thread_data.ctxt.mp


class _Unspecified:
    def __repr__(self):
        return "unspecified"


_UNSPECIFIED = _Unspecified()


@typing.overload
def compute_context(
    mesh: Optional[Mesh], axis_mapping: Optional[ResourceMapping] = None, mp: Optional[jmp.Policy] = None
) -> "ComputeContext":
    ...


@typing.overload
def compute_context(*, axis_mapping: Optional[ResourceMapping], mp: Optional[jmp.Policy] = None) -> "ComputeContext":
    ...


@typing.overload
def compute_context(*, mp: Optional[jmp.Policy]) -> "ComputeContext":
    ...


@typing.overload
def compute_context() -> AbstractContextManager["ComputeContext"]:
    ...


def compute_context(
    mesh: Optional[Mesh] | _Unspecified = _UNSPECIFIED,
    axis_mapping: Optional[ResourceMapping] = None,
    mp: Optional[jmp.Policy] = None,
):

    if isinstance(mesh, _Unspecified) and axis_mapping is None and mp is None:
        return _mapping_holder.thread_data.ctxt

    if mesh is None or isinstance(mesh, _Unspecified):
        mesh = haliax.partitioning._get_mesh()

    if axis_mapping is None:
        axis_mapping = haliax.partitioning.current_thread_local_mapping()

    if mp is None:
        mp = _mapping_holder.thread_data.ctxt.mp

    ctxt = ComputeContext(mesh, axis_mapping, mp)
    return ctxt


class ComputeContext(AbstractContextManager):
    def __init__(self, mesh: Mesh, axis_mapping: ResourceMapping, mp: jmp.Policy):
        self.mesh = mesh
        self.axis_mapping = axis_mapping
        self.mp = mp

        self._context_managers: Optional[contextlib.ExitStack] = None
        self._old_ctxt: Optional[ComputeContext] = None

    def __enter__(self):
        self._old_ctxt = _mapping_holder.thread_data.ctxt
        self._context_managers = contextlib.ExitStack()
        self._context_managers.__enter__()
        self._context_managers.enter_context(self.mesh)
        self._context_managers.enter_context(haliax.axis_mapping(self.axis_mapping))
        _mapping_holder.thread_data.ctxt = self
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        try:
            if self._context_managers is None:
                raise RuntimeError("context manager is not entered")

            self._context_managers.__exit__(exc_type, exc_value, traceback)
        finally:
            self._context_managers = None
            _mapping_holder.thread_data.ctxt = self._old_ctxt
            self._old_ctxt = None


class _ComputeContextManagerHolder:
    """Global holder for compute context manager."""

    def __init__(self):
        self.thread_data = threading.local()
        self.thread_data.ctxt = None


_mapping_holder = _ComputeContextManagerHolder()
