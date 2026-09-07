import numpy as np

from shadlabutils.intanutil.data import initialize_memory, advance_index, read_timestamps, read_digital_signals


def read_analog_signal_type(fid, dest, start, num_samples, num_channels):
    """Reads data from binary file as a NumPy array, indexing them into
    'dest', which should be an analog signal type within 'data', for example
    data['amplifier_data'] or data['aux_input_data']. Each sample is assumed
    to be of dtype 'uint16'.
    """

    if num_channels < 1:
        return
    end = start + num_samples
    tmp = np.fromfile(fid, dtype='uint16', count=num_samples*num_channels)

def read_analog_signals(fid, data, index, samples_per_block, header):
    """Reads all analog signal types present in RHD files: amplifier_data,
    aux_input_data, supply_voltage_data, temp_sensor_data, and board_adc_data,
    into 'data' dict.
    """

    read_analog_signal_type(fid,
                            data['amplifier_data'],
                            index,
                            samples_per_block,
                            header['num_amplifier_channels'])

    if header['dc_amplifier_data_saved']:
        read_analog_signal_type(fid,
                                data['dc_amplifier_data'],
                                index,
                                samples_per_block,
                                header['num_amplifier_channels'])

    read_analog_signal_type(fid,
                            data['stim_data_raw'],
                            index,
                            samples_per_block,
                            header['num_amplifier_channels'])

    read_analog_signal_type(fid,
                            data['board_adc_data'],
                            index,
                            samples_per_block,
                            header['num_board_adc_channels'])

    read_analog_signal_type(fid,
                            data['board_dac_data'],
                            index,
                            samples_per_block,
                            header['num_board_dac_channels'])

def read_one_data_block(data, header, index, fid):
    """Reads one 60 or 128 sample data block from fid into data,
    at the location indicated by index."""
    samples_per_block = header['num_samples_per_data_block']

    read_timestamps(fid,
                    data,
                    index,
                    samples_per_block)

    read_analog_signals(fid,
                        data,
                        index,
                        samples_per_block,
                        header)

    read_digital_signals(fid,
                         data,
                         index,
                         samples_per_block,
                         header)

def read_all_data_blocks(header, num_samples, num_blocks, fid):
    """Reads all data blocks present in file, allocating memory for and
    returning 'data' dict containing all data.
    """
    data, index = initialize_memory(header, num_samples)
    for i in range(num_blocks):
        read_one_data_block(data, header, index, fid)
        index = advance_index(index, header['num_samples_per_data_block'])
    return data