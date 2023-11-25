from _typeshed import HasFileno, OptExcInfo, ReadOnlyBuffer
from _typeshed.wsgi import WSGIApplication
from collections.abc import Callable
from types import ModuleType
from typing import Any, Protocol, overload
from typing_extensions import Literal, Self, TypeAlias, final

import uwsgidecorators

_TrueOrNone: TypeAlias = Literal[True] | None

class _RPCCallable(Protocol):
    def __call__(self, *args: bytes) -> bytes | None: ...

# FIXME: Technically we know the exact layout of _AppsDict and _WorkerDict
#        but TypedDict does not support bytes keys, so for now we use type
#        aliases to a more generic dict
_WorkerDict: TypeAlias = dict[bytes, Any]

SPOOL_IGNORE: Literal[0]
SPOOL_OK: Literal[-2]
SPOOL_RETRY: Literal[-1]
applications: dict[str, WSGIApplication | str] | None
buffer_size: int
cores: int
has_threads: int
hostname: bytes
is_a_reload: bool
loop: bytes | None
magic_table: dict[bytes, bytes]
numproc: int
opt: dict[str, bytes | Literal[True] | list[bytes | Literal[True]]]
sockets: list[int]
started_on: int
unbit: _TrueOrNone
version: bytes
version_info: tuple[int, int, int, int, bytes]
spoolers: tuple[bytes, ...]
queue_size: int

decorators = uwsgidecorators
spooler = uwsgidecorators.manage_spool_request
post_fork_hook = uwsgidecorators.postfork_chain_hook

@final
class SymbolsImporter:
    def find_module(self, __fullname: str) -> Self | None: ...
    def load_module(self, __fullname: str) -> ModuleType | None: ...

@final
class SymbolsZipImporter:
    def __init__(self, __name: str) -> None: ...
    def find_module(self, __fullname: str) -> Self | None: ...
    def load_module(self, __fullname: str) -> ModuleType | None: ...

@final
class ZipImporter:
    def __init__(self, __name: str) -> None: ...
    def find_module(self, __fullname: str) -> Self | None: ...
    def load_module(self, __fullname: str) -> ModuleType | None: ...

def accepting(__accepting: bool = True) -> None: ...
def add_cron(__signum: int, __minute: int, __hour: int, __day: int, __month: int, __weekday: int) -> Literal[True]: ...
def add_file_monitor(__signum: int, __filename: str) -> None: ...
def add_rb_timer(__signum: int, __seconds: int, __iterations: int = 0) -> None: ...
def add_timer(__signum: int, __seconds: int) -> None: ...
def add_var(__key: bytes | str, __val: bytes | str) -> Literal[True]: ...
def alarm(__alarm: str, __msg: bytes | ReadOnlyBuffer | str) -> None: ...
def async_connect(__socket_name: str) -> int: ...
def async_sleep(__timeout: float) -> Literal[b""]: ...
def cache_clear(__cache_name: str = ...) -> _TrueOrNone: ...
def cache_dec(__key: str | bytes, __decrement: int = 1, __expires: int = 0, __cache_name: str = ...) -> _TrueOrNone: ...
def cache_del(__key: str | bytes, __cache_name: str = ...) -> _TrueOrNone: ...
def cache_div(__key: str | bytes, __divisor: int = 2, __expires: int = 0, __cache_name: str = ...) -> _TrueOrNone: ...
def cache_exists(__key: str | bytes, __cache_name: str = ...) -> _TrueOrNone: ...
def cache_get(__key: str | bytes, __cache_name: str = ...) -> bytes | None: ...
def cache_inc(__key: str | bytes, __increment: int = 1, __expires: int = 0, __cache_name: str = ...) -> _TrueOrNone: ...
def cache_keys(__cache_name: str = ...) -> list[bytes]: ...
def cache_mul(__key: str | bytes, __factor: int = 2, __expires: int = 0, __cache_name: str = ...) -> _TrueOrNone: ...
def cache_num(__key: str | bytes, __cache_name: str = ...) -> int: ...
def cache_set(
    __key: str | bytes, __value: str | bytes | ReadOnlyBuffer, __expires: int = 0, __cache_name: str = ...
) -> _TrueOrNone: ...
def cache_update(
    __key: str | bytes, __value: str | bytes | ReadOnlyBuffer, __expires: int = 0, __cache_name: str = ...
) -> _TrueOrNone: ...
def queue_get(__index: int) -> bytes | None: ...
def queue_set(__index: int, __message: str | bytes | ReadOnlyBuffer) -> _TrueOrNone: ...
@overload
def queue_last(__num: Literal[0] = 0) -> bytes | None: ...  # type:ignore[misc]
@overload
def queue_last(__num: int) -> list[bytes | None]: ...
def queue_push(__message: str | bytes | ReadOnlyBuffer) -> _TrueOrNone: ...
def queue_pull() -> bytes | None: ...
def queue_pop() -> bytes | None: ...
def queue_slot() -> int: ...
def queue_pull_slot() -> int: ...
def snmp_set_community(__snmp_community: str) -> Literal[True]: ...
def snmp_set_counter32(__oid_num: int, __value: int) -> _TrueOrNone: ...
def snmp_set_counter64(__oid_num: int, __value: int) -> _TrueOrNone: ...
def snmp_set_gauge(__oid_num: int, __value: int) -> _TrueOrNone: ...
def snmp_incr_counter32(__oid_num: int, __increment: int) -> _TrueOrNone: ...
def snmp_incr_counter64(__oid_num: int, __increment: int) -> _TrueOrNone: ...
def snmp_incr_gauge(__oid_num: int, __increment: int) -> _TrueOrNone: ...
def snmp_decr_counter32(__oid_num: int, __decrement: int) -> _TrueOrNone: ...
def snmp_decr_counter64(__oid_num: int, __decrement: int) -> _TrueOrNone: ...
def snmp_decr_gauge(__oid_num: int, __decrement: int) -> _TrueOrNone: ...
@overload
def send_to_spooler(__mesage_dict: dict[bytes, bytes]) -> bytes | None: ...
@overload
def send_to_spooler(
    *, spooler: bytes = ..., priority: bytes = ..., at: bytes = ..., body: bytes = ..., **kwargs: bytes
) -> bytes | None: ...

spool = send_to_spooler

def set_spooler_frequency(__frequency: int) -> Literal[True]: ...
def spooler_jobs() -> list[bytes]: ...
def spooler_pid() -> int: ...
def spooler_pids() -> list[int]: ...
def spooler_get_task(__task_path: str) -> dict[bytes, bytes] | None: ...
def call(__rpc_name: str, *args: bytes) -> bytes | None: ...
def chunked_read(__timeout: int = 0) -> bytes: ...
def chunked_read_nb() -> bytes: ...
def cl() -> int: ...
def close(__fd: int) -> None: ...
def connect(__socket_name: str, timeout: int = 0) -> int: ...
def connection_fd() -> int: ...
def disconnect() -> None: ...
def embedded_data(__name: str) -> bytes: ...
def extract(__name: str) -> bytes | None: ...
def farm_get_msg() -> bytes | None: ...
def farm_msg(__farm_name: str, __message: str | bytes | ReadOnlyBuffer) -> None: ...
def get_logvar(__key: str | bytes) -> bytes | None: ...
def green_schedule() -> Literal[True]: ...
def i_am_the_lord(__legion_name: str) -> bool: ...
def i_am_the_spooler() -> _TrueOrNone: ...
def in_farm(__farm_name: str = ...) -> _TrueOrNone: ...
def is_connected(__fd: int) -> bool: ...
def is_locked(__lock_num: int = 0) -> bool: ...
def listen_queue(__id: int = 0) -> int: ...
def lock(__lock_num: int = 0) -> None: ...
def log(__logline: str) -> Literal[True]: ...
def log_this_request() -> None: ...
def logsize() -> int: ...
def lord_scroll(__legion_name: str) -> bytes | None: ...
def masterpid() -> int: ...
def mem() -> tuple[int, int]: ...
def metric_dec(__key: str, __decrement: int = 1) -> _TrueOrNone: ...
def metric_div(__key: str, __divisor: int = 1) -> _TrueOrNone: ...
def metric_get(__key: str) -> int: ...
def metric_inc(__key: str, __increment: int = 1) -> _TrueOrNone: ...
def metric_mul(__key: str, __factor: int = 1) -> _TrueOrNone: ...
def metric_set(__key: str, __value: int = 1) -> _TrueOrNone: ...
def metric_set_max(__key: str, __value: int = 1) -> _TrueOrNone: ...
def metric_set_min(__key: str, __value: int = 1) -> _TrueOrNone: ...
def micros() -> int: ...
def mule_get_msg(signals: bool = True, farms: bool = True, buffer_size: int = 65536, timeout: int = -1) -> bytes: ...
def mule_id() -> int: ...
@overload
def mule_msg(__mesage: str | bytes | ReadOnlyBuffer) -> bool: ...
@overload
def mule_msg(__mesage: str | bytes | ReadOnlyBuffer, __mule_id: int) -> bool: ...
@overload
def mule_msg(__mesage: str | bytes | ReadOnlyBuffer, __farm_name: str) -> bool: ...
def offload(__filename: str, __len: int = 0) -> Literal[b""]: ...
def parsefile(__filename: str) -> dict[bytes, bytes] | None: ...
def ready() -> _TrueOrNone: ...
def ready_fd() -> int: ...
def recv(__fd: int, __max_size: int = 4096) -> bytes | None: ...
@overload
def register_rpc(__name: str, __func: Callable[[], bytes | None]) -> Literal[True]: ...
@overload
def register_rpc(__name: str, __func: Callable[[bytes], bytes | None], arg_count: Literal[1]) -> Literal[True]: ...
@overload
def register_rpc(__name: str, __func: Callable[[bytes, bytes], bytes | None], arg_count: Literal[2]) -> Literal[True]: ...
@overload
def register_rpc(__name: str, __func: _RPCCallable, arg_count: int) -> Literal[True]: ...
def register_signal(__signum: int, __who: str, __handler: Callable[[int], Any]) -> None: ...
def reload() -> _TrueOrNone: ...
def request_id() -> int: ...
def route(__router_name: str, __router_args: str) -> int: ...
def rpc(__node: str | bytes, __rpc_name: bytes, *rpc_args: bytes) -> bytes | None: ...
def rpc_list() -> tuple[bytes, ...]: ...
def scrolls(__legion_name: str) -> list[bytes] | None: ...
@overload
def send(__data: bytes) -> _TrueOrNone: ...
@overload
def send(__fd: int, __data: bytes) -> _TrueOrNone: ...
def sendfile(
    __filename_or_fd: str | bytes | int | HasFileno, __chunk: int = 0, __pos: int = 0, filesize: int = 0
) -> _TrueOrNone: ...
def set_logvar(__key: str | bytes, __val: str | bytes) -> None: ...
def set_user_harakiri(__seconds: int) -> None: ...
def set_warning_message(__message: str) -> Literal[True]: ...
def setprocname(__name: str) -> None: ...
def signal(__signum: int = ..., __node: str = ...) -> None: ...
def signal_received() -> int: ...
def signal_registered(__signum: int) -> _TrueOrNone: ...
def signal_wait(__signum: int = ...) -> Literal[b""]: ...
def start_response(
    __status: str, __headers: list[tuple[str, str]], __exc_info: OptExcInfo | None = ...
) -> Callable[[bytes], None]: ...
def stop() -> _TrueOrNone: ...
def suspend() -> Literal[True]: ...
def total_requests() -> int: ...
def unlock(__lock_num: int = 0) -> None: ...
def wait_fd_read(__fd: int, __timeout: int = 0) -> Literal[b""]: ...
def wait_fd_write(__fd: int, __timeout: int = 0) -> Literal[b""]: ...
def websocket_handshake(__key: str | bytes = ..., __origin: str | bytes = ..., __proto: str | bytes = ...) -> None: ...
def websocket_recv() -> bytes: ...
def websocket_recv_nb() -> bytes: ...
def websocket_send(message: str | bytes | ReadOnlyBuffer) -> None: ...
def websocket_send_binary(message: str | bytes | ReadOnlyBuffer) -> None: ...
def worker_id() -> int: ...
def workers() -> tuple[_WorkerDict, ...] | None: ...
def sharedarea_read(__id: int, __position: int, __length: int) -> bytes: ...
def sharedarea_write(__id: int, __position: int, __value: str | bytes | ReadOnlyBuffer) -> None: ...
def sharedarea_readbyte(__id: int, __position: int) -> int: ...
def sharedarea_writebyte(__id: int, __position: int, __value: int) -> None: ...
def sharedarea_read8(__id: int, __position: int) -> int: ...
def sharedarea_write8(__id: int, __position: int, __value: int) -> None: ...
def sharedarea_readlong(__id: int, __position: int) -> int: ...
def sharedarea_writelong(__id: int, __position: int, __value: int) -> None: ...
def sharedarea_read64(__id: int, __position: int) -> int: ...
def sharedarea_write64(__id: int, __position: int, __value: int) -> None: ...
def sharedarea_read32(__id: int, __position: int) -> int: ...
def sharedarea_write32(__id: int, __position: int, __value: int) -> None: ...
def sharedarea_read16(__id: int, __position: int) -> int: ...
def sharedarea_write16(__id: int, __position: int, __value: int) -> None: ...
def sharedarea_inclong(__id: int, __position: int, __increment: int = 1) -> None: ...
def sharedarea_inc64(__id: int, __position: int, __increment: int = 1) -> None: ...
def sharedarea_inc32(__id: int, __position: int, __increment: int = 1) -> None: ...
def sharedarea_dec64(__id: int, __position: int, __decrement: int = 1) -> None: ...
def sharedarea_dec32(__id: int, __position: int, __decrement: int = 1) -> None: ...
def sharedarea_rlock(__id: int) -> None: ...
def sharedarea_wlock(__id: int) -> None: ...
def sharedarea_unlock(__id: int) -> None: ...
def sharedarea_object(__id: int) -> bytearray: ...
def sharedarea_memoryview(__id: int) -> memoryview: ...
