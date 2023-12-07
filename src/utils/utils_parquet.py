
import pyarrow.parquet as pq

def get_dataset_info(path):
    dataset = pq.ParquetDataset(path)
    total_rows = 0
    total_byte_size = 0
    for piece in dataset.pieces:
        metadata = piece.get_metadata()
        total_rows += metadata.num_rows
        for row_group_index in range(metadata.num_row_groups):
            row_group = metadata.row_group(row_group_index)
            total_byte_size += row_group.total_byte_size

    if total_rows == 0:
        raise ValueError('No rows found in dataset: {}'.format(path))

    if total_byte_size == 0:
        raise ValueError('No data found in dataset: {}'.format(path))

    if total_rows > total_byte_size:
        raise ValueError('Found {} bytes in {} rows;  dataset may be corrupted.'
                         .format(total_byte_size, total_rows))

    return total_rows, total_byte_size