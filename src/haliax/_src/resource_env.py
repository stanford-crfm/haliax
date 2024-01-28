import threading
from contextlib import AbstractContextManager
from typing import Optional

import jmp
from jax.sharding import Mesh

import haliax
from haliax.types import ResourceMapping


def current_resource_env() -> "ResourceEnv":
    cur = _context_holder.thread_data.ctxt

    # the mesh can change in JAX without us knowing, so we need to check
    mesh = _get_mesh()
    if _get_mesh() is not cur.mesh:
        cur = cur.with_mesh(mesh)

    return cur


DEFAULT_MP_POLICY = jmp.get_policy("f32")


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
        mp = _context_holder.thread_data.ctxt.mp

    if mp is None:
        mp = DEFAULT_MP_POLICY

    ctxt = ResourceEnv(axis_mapping, mp, mesh)
    return ctxt


class ResourceEnv(AbstractContextManager):
    """
    A ResourceEnv is a context manager that can be used to specify the mesh, axis mapping, and mixed-precision policy to
    use for computation. It can be used as a context manager or just passed to a function as an argument.

    It should be noted that JAX internals has a ResourceEnv that sort of does a similar thing (minus the mixed-precision
    policy). However, it is not exposed to the user, and its semantic axes are kind of deprecated.
    """

    def __init__(self, axis_mapping: Optional[ResourceMapping], mp: jmp.Policy | str, mesh: Optional[Mesh]):
        self.mesh = mesh
        self.axis_mapping = axis_mapping
        if isinstance(mp, str):
            mp = jmp.get_policy(mp)
        self.mp = mp

    def with_policy(self, mp: jmp.Policy) -> "ResourceEnv":
        return ResourceEnv(self.axis_mapping, mp, self.mesh)

    def with_mesh(self, mesh: Mesh) -> "ResourceEnv":
        return ResourceEnv(self.axis_mapping, self.mp, mesh)

    def with_axis_mapping(self, axis_mapping: ResourceMapping) -> "ResourceEnv":
        return ResourceEnv(axis_mapping, self.mp, self.mesh)

    def __enter__(self):
        _context_holder.thread_data.ctxt = self
        _context_holder.thread_data.stack.append(self)

        if self.mesh:
            self.mesh.__enter__()

        return self

    def __exit__(self, exc_type, exc_value, traceback):
        _context_holder.thread_data.stack.pop()
        _context_holder.thread_data.ctxt = _context_holder.thread_data.stack[-1]

        if self.mesh:
            self.mesh.__exit__(exc_type, exc_value, traceback)


class _ComputeContextManagerHolder:
    """Global holder for compute context manager."""

    def __init__(self):
        self.thread_data = threading.local()
        self.thread_data.ctxt = ResourceEnv(None, DEFAULT_MP_POLICY, None)
        self.thread_data.stack = []
        self.thread_data.stack.append(self.thread_data.ctxt)


_context_holder = _ComputeContextManagerHolder()


def _get_mesh():
    from jax.experimental.maps import thread_resources

    return thread_resources.env.physical_mesh
