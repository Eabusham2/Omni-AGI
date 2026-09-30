"""Pre-decode resource admission for Arrow/Parquet native buffers.

Metadata estimates are conservative admission, not an allocator hard cap.
See Apache Arrow format/Message.fbs for IPC buffer compression prefixes.
"""

import struct
from pathlib import Path
from .text_spool import DatasetResourcePause, require_parser_resources


def admit_parquet_footer(path):
    size = Path(path).stat().st_size
    with open(path, "rb") as source:
        if size < 12:
            raise ValueError("Parquet footer is truncated")
        source.seek(size - 8)
        footer_size, magic = struct.unpack("<I4s", source.read(8))
    if magic != b"PAR1" or footer_size > size - 12:
        raise ValueError("Parquet footer declaration is invalid")
    require_parser_resources("Parquet footer decode", ram_bytes=footer_size * 8)


def admit_parquet_row_group(metadata, index):
    group = metadata.row_group(index)
    total = int(group.total_byte_size)
    if total < 0:
        raise ValueError("Parquet row group byte declaration is invalid")
    # Native pages/dictionaries may outlive a one-row batch. Do not pretend
    # batch_size=1 is a byte bound: admit the complete declared row group.
    require_parser_resources("Parquet native row-group decode", ram_bytes=total * 3)


def _field(data, table, slot):
    vtable = table - struct.unpack_from("<i", data, table)[0]
    length = struct.unpack_from("<H", data, vtable)[0]
    entry = vtable + 4 + slot * 2
    if entry + 2 > vtable + length:
        return None
    offset = struct.unpack_from("<H", data, entry)[0]
    return table + offset if offset else None


def _reference(data, field):
    return field + struct.unpack_from("<I", data, field)[0]


def ipc_message_allocation(message):
    """Read small metadata and 8-byte prefixes, never copy/decompress a body."""
    metadata = memoryview(message.metadata)
    body = memoryview(message.body) if message.body is not None else memoryview(b"")
    root = struct.unpack_from("<I", metadata, 0)[0]
    type_field = _field(metadata, root, 1)
    header_kind = metadata[type_field] if type_field is not None else 0
    if header_kind not in {2, 3}:
        return header_kind, len(metadata) * 8
    header_field = _field(metadata, root, 2)
    if header_field is None:
        raise ValueError("Arrow message has no record-batch header")
    batch = _reference(metadata, header_field)
    if header_kind == 2:
        dictionary_data = _field(metadata, batch, 1)
        if dictionary_data is None:
            raise ValueError("Arrow dictionary message has no data")
        batch = _reference(metadata, dictionary_data)
    buffers_field = _field(metadata, batch, 2)
    if buffers_field is None:
        return header_kind, len(metadata) * 8
    vector = _reference(metadata, buffers_field)
    count = struct.unpack_from("<I", metadata, vector)[0]
    compression_field = _field(metadata, batch, 3)
    compressed = compression_field is not None
    if compressed:
        compression = _reference(metadata, compression_field)
        method_field = _field(metadata, compression, 1)
        if method_field is not None and metadata[method_field] != 0:
            raise DatasetResourcePause("Arrow IPC compression method has no verified allocation admission")
    allocation = 0
    for index in range(count):
        offset, length = struct.unpack_from("<qq", metadata, vector + 4 + index * 16)
        if offset < 0 or length < 0 or offset + length > len(body):
            raise ValueError("Arrow buffer range is invalid")
        decoded = length
        if compressed and length:
            if length < 8:
                raise ValueError("Arrow compressed buffer has no length prefix")
            decoded = struct.unpack_from("<q", body, offset)[0]
            if decoded == -1:
                decoded = length - 8
            elif decoded < 0:
                raise ValueError("Arrow uncompressed buffer length is invalid")
        allocation += decoded
    return header_kind, allocation * 3 + len(metadata) * 8


def ipc_allocation_frames(source, ipc, file_size):
    """Yield batch admission estimates from a separate memory-mapped source."""
    source.seek(0)
    magic = source.read(6)
    if magic == b"ARROW1":
        source.seek(file_size - 10)
        footer = source.read(10)
        footer_size = struct.unpack_from("<I", footer, 0)[0]
        if footer[4:] != b"ARROW1" or footer_size > file_size - 18:
            raise ValueError("Arrow file footer is invalid")
        require_parser_resources("Arrow IPC footer decode", ram_bytes=footer_size * 8)
        position, stop = 8, file_size - 10 - footer_size
    else:
        position, stop = 0, file_size
    while position < stop:
        source.seek(position)
        prefix = source.read(4)
        if len(prefix) != 4:
            raise ValueError("Arrow IPC frame prefix is truncated")
        metadata_size = struct.unpack("<I", prefix)[0]
        if metadata_size == 0xFFFFFFFF:
            continuation = source.read(4)
            if len(continuation) != 4:
                raise ValueError("Arrow IPC continuation is truncated")
            metadata_size = struct.unpack("<I", continuation)[0]
        if not metadata_size:
            break
        if source.tell() + metadata_size > stop:
            raise ValueError("Arrow IPC frame metadata exceeds the file")
        require_parser_resources("Arrow IPC message metadata", ram_bytes=metadata_size * 8)
        source.seek(position)
        message = ipc.read_message(source)
        following = source.tell()
        if following <= position or following > stop:
            raise ValueError("Arrow IPC frame body exceeds the file")
        try:
            yield ipc_message_allocation(message)
        except (IndexError, struct.error) as error:
            raise ValueError("Arrow IPC allocation metadata is truncated") from error
        position = following
