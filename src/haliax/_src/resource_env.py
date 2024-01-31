import threading
from contextlib import AbstractContextManager
from typing import Optional

import jmp
from jax.sharding import Mesh

import haliax
from haliax.types import ResourceMapping


def current_resource_env() -> "ResourceEnv":
    cur = _context_holder.get_env()

    # the mesh can change in JAX without us knowing, so we need to check
    mesh = _get_mesh()
    if _get_mesh() is not cur.mesh:
        cur = cur.with_mesh(mesh)

    return cur


def resource_env(
    axis_mapping: Optional[ResourceMapping] = None, mp: Optional[jmp.Policy | str] = None, mesh: Optional[Mesh] = None
) -> "ResourceEnv":
    """
    When called with arguments, returns a compute context env that can be used in a `with` statement.
    Args:
        mesh: mesh to use in the context
        axis_mapping: axis mapping to use in the context
        mp: mixed-precision policy to use in the context

    Returns:
        A ResourceEnv object that can be used as a context manager.
    """

    if mesh is None:
        mesh = _get_mesh()

    if axis_mapping is None:
        axis_mapping = haliax.partitioning.current_mapping()

    if mp is None:
        mp = haliax.current_mp_policy()

    return ResourceEnv(axis_mapping, mp, mesh)


class ResourceEnv(AbstractContextManager):
    """
    A ResourceEnv is a context manager that can be used to specify the mesh, axis mapping, and mixed-precision policy to
    use for computation. It can be used as a context manager or just passed to a function as an argument.

    It should be noted that JAX internals has a ResourceEnv that sort of does a similar thing (minus the mixed-precision
    policy). However, it is not exposed to the user, and its semantic axes are kind of deprecated.
    """

    def __init__(self, axis_mapping: Optional[ResourceMapping], mp: Optional[jmp.Policy | str], mesh: Optional[Mesh]):
        self.mesh = mesh
        self.axis_mapping = axis_mapping
        if isinstance(mp, str):
            mp = jmp.get_policy(mp)
        self.mp = mp

    def with_policy(self, mp: Optional[jmp.Policy | str]) -> "ResourceEnv":
        return ResourceEnv(self.axis_mapping, mp, self.mesh)

    def with_mesh(self, mesh: Mesh) -> "ResourceEnv":
        return ResourceEnv(self.axis_mapping, self.mp, mesh)

    def with_axis_mapping(self, axis_mapping: ResourceMapping) -> "ResourceEnv":
        return ResourceEnv(axis_mapping, self.mp, self.mesh)

    def __enter__(self):
        _context_holder.push_env(self)

        if self.mesh:
            self.mesh.__enter__()

        return self

    def __exit__(self, exc_type, exc_value, traceback):
        _context_holder.pop_env()

        if self.mesh:
            self.mesh.__exit__(exc_type, exc_value, traceback)


class _ComputeContextManagerHolder:
    """Global holder for compute context manager."""

    def __init__(self):
        self.thread_data = threading.local()
        self.thread_data.stack = [DEFAULT_RESOURCE_ENV]

    def push_env(self, ctxt: ResourceEnv):
        if not hasattr(self.thread_data, "stack"):
            self.thread_data.stack = [DEFAULT_RESOURCE_ENV]
        self.thread_data.stack.append(ctxt)

    def get_env(self) -> ResourceEnv:
        if not hasattr(self.thread_data, "stack"):
            return DEFAULT_RESOURCE_ENV
        return self.thread_data.stack[-1]

    def pop_env(self):
        if not hasattr(self.thread_data, "stack"):
            raise ValueError("No context to pop.")
        self.thread_data.stack.pop()

        return self.get_env()


DEFAULT_RESOURCE_ENV = ResourceEnv(None, None, None)
_context_holder = _ComputeContextManagerHolder()


def _get_mesh():
    from jax.experimental.maps import thread_resources

    return thread_resources.env.physical_mesh
