import gzip
import json
import os
import csv
import numpy as np
import pandas as pd

from collections import Counter

micro_to_minutes = 1.66666667 * 10 ** (-8)
micro_to_milliseconds = 0.001
micro_to_seconds = 0.000001


# def print_profiler_results(path_to_logs: str):
#
#     for root, dirs, _ in os.walk(path_to_logs + '/plugins/profile/'):
#         for d in dirs:
#             directory = os.path.join(root, d)
#             print('TTETEETETET')
#             print(directory)
#
#     my_items = os.listdir(directory + '/')
#     print(any(".trace.json.gz" in s for s in my_items))
#
#     matching = [s for s in my_items if ".trace.json.gz" in s]
#
#     print(directory)
#     with gzip.open(directory + '/' + matching[0], "r") as f:
#         data = f.read()
#         j = json.loads(data.decode('utf-8'))
#
#         micro_to_minutes = 1.66666667 * 10 ** (-8)
#         micro_to_milliseconds = 0.001
#         micro_to_seconds = 0.000001
#         my_dict = {}
#         occurrences = []
#
#         for i in range(len(j["traceEvents"])):
#
#             if 'name' in j["traceEvents"][i] and j["traceEvents"][i].get("name") not in my_dict.keys():
#                 # add key, value pair to dictionary
#                 my_dict[j["traceEvents"][i].get("name")] = 0.0
#
#             if 'dur' in j["traceEvents"][i].keys():
#                 my_dict[j["traceEvents"][i].get("name")] += j["traceEvents"][i].get("dur")
#
#             if 'name' in j["traceEvents"][i].keys():
#                 occurrences.append(j["traceEvents"][i].get("name"))
#
#             else:
#                 continue
#
#         f = open(path_to_logs + '/output_data.csv', 'w')
#         writer = csv.writer(f)
#
#         writer.writerow(['name', 'time in milliseconds', 'ratio', 'occurrences'])
#
#         total = 0.0
#         print(my_dict)
#
#         # total time of running the model
#         # print(runtimes)
#         # sorted_runtimes = sorted(runtimes)
#         # print(sorted_runtimes)
#         print("-" * 25, "TOTAL TIME", '-' * 25, '\n')
#
#         for value in my_dict.values():
#             total += value
#         print(f'total time in seconds per inference: {total * micro_to_seconds}')
#         print('\n' * 3)
#
#         occurrences_counter = Counter(occurrences)
#         # ratio and total time par layer
#         print("-" * 25, "RATIOS AND TIME PER LAYER", '-' * 25, '\n')
#         for pair in my_dict.items():
#             in_milliseconds = pair[1] * micro_to_milliseconds
#             ratio = pair[1] / total
#             occurrences = occurrences_counter.get(pair[0])
#             print(f'for {pair[0]} total time in milliseconds is: {in_milliseconds}')
#             print(f'for {pair[0]} ratio to total time is: {pair[1] / total}')
#             print(f'{pair[0]} has occurred {occurrences} times')
#             print('*' * 45)
#             row = [pair[0], in_milliseconds, ratio, occurrences]
#             writer.writerow(row)
#         print('\n' * 3)
#
#         f.close()

def calculate_total_time(events_df):
    events_df = events_df.sort_values(by="Start")
    end = -1

    for index, row in events_df.iterrows():
        func_end_time = row["Start"] + row["Duration"]
        if func_end_time > end:
            end = func_end_time
    return end


def variable_read_op_time(events_df):
    return events_df.loc[events_df['Op_Name'] == "ReadVariableOp"]["Duration"].sum()


def print_profiler_results_dls_only(dls_opts_df, total):
    console = Console()
    res_table = Table(show_header=True, header_style="bold magenta", title="TF Profiler results - DLS ops")
    res_table.add_column("DLS Layer", width=15)
    res_table.add_column("Total time ms", width=10)
    res_table.add_column("Ratio of total time %", width=5)
    res_table.add_column("Shape", width=70)
    res_table.add_column("Number of occurences", width=5)

    for index, row in dls_opts_df.iterrows():
        name = row["Op_Name"].replace(":_DLSSuperNode", '').replace("DLS-SN-", '')
        in_milliseconds = row["Duration"] * micro_to_milliseconds
        ratio = row["Duration"] / total * 100
        res_table.add_row(name, "{:.4f}".format(in_milliseconds), "{:.2f}".format(ratio), row["Shape"],
                          str(row["Occurences"]))
    console.print(res_table)


def print_profiler_results_all(events_df, total):
    console = Console()
    res_table = Table(show_header=True, header_style="bold magenta", title="TF Profiler results - All ops")
    res_table.add_column("DLS Layer", width=15)
    res_table.add_column("Total time ms", width=10)
    res_table.add_column("Ratio of total time %", width=5)
    res_table.add_column("Shape", width=70)
    res_table.add_column("Number of occurences", width=5)

    for index, row in events_df.iterrows():
        name = row["Op_Name"].replace(":_DLSSuperNode", '').replace("DLS-SN-", '')
        in_milliseconds = row["Duration"] * micro_to_milliseconds
        ratio = row["Duration"] / total * 100
        res_table.add_row(name, "{:.4f}".format(in_milliseconds), "{:.2f}".format(ratio), row["Shape"],
                          str(row["Occurences"]))
    console.print(res_table)


def print_total_vs_dls_time(total, dls_time_sum, variable_read_ops):
    console = Console()
    table = Table(show_header=True, header_style="bold magenta")
    table.add_column("Total time", width=45, justify="center")
    table.add_column("DLS time", width=45, justify="center")
    table.add_column("DLS to Total ratio %", width=8, justify="right")
    table.add_column("DLS to Total ratio excle varReadOP %", width=11, justify="right")
    ratio = dls_time_sum / total * 100
    ratio_excl_varReadOP = dls_time_sum / (total - variable_read_ops) * 100
    table.add_row("{:.5f}".format(total * micro_to_seconds), "{:.5f}".format(dls_time_sum * micro_to_seconds),
                  "{:.2f}".format(ratio), "{:.2f}".format(ratio_excl_varReadOP))
    console.print(table)


def print_csv_logfile_location(directory):
    console = Console()
    table = Table(show_header=True, header_style="bold magenta")
    table.add_column("CSV log file stored under adress:", width=118, justify="center")
    table.add_row(directory)
    console.print(table)


def generate_otput_csv_file(events_df, dls_opts_df, path_to_logs):
    f = open(path_to_logs + '/output_data.csv', 'w')
    writer = csv.writer(f)
    writer.writerow(['name', 'time in milliseconds', 'ratio', 'occurrences'])
    # events_df.to_csv(path_to_logs + '/output_data_df.csv', index = False)
    # TODO MARCEL add business logic
    f.close()


def print_profiler_results(path_to_logs: str):
    for root, dirs, _ in os.walk(path_to_logs + '/plugins/profile/'):
        for d in dirs:
            directory = os.path.join(root, d)

    print_csv_logfile_location(directory)

    my_items = os.listdir(directory + '/')
    matching = [s for s in my_items if ".trace.json.gz" in s]
    json_path = directory + '/' + matching[-1]

    with gzip.open(json_path, "r") as f:
        data = f.read()
        json_data = json.loads(data.decode('utf-8'))

        data_columns = ["Op_Name", "Start", "Duration", "Occurences", "Shape", "Is_DLS_Op"]
        events_df = pd.DataFrame(columns=data_columns)

        for i in range(len(json_data["traceEvents"])):
            name = ""
            start = -1.0
            duration = -1.0
            shape = str("")
            is_DLS_Op = False
            occurences = 0

            if 'name' in json_data["traceEvents"][i]:
                tmp_name = json_data["traceEvents"][i].get("name")
                if tmp_name not in events_df["Op_Name"]:
                    name = tmp_name
                    occurences = 1
                    if json_data["traceEvents"][i]["name"].find("DLS") != -1:
                        is_DLS_Op = True
                else:
                    occurences_val = events_df.loc[events_df['Op_Name'] == tmp_name]["occurences"]
                    occurences_val = occurences_val + 1
                    events_df.loc[events_df['Op_Name'] == tmp_name]["occurences"] = occurences_val

            if 'dur' in json_data["traceEvents"][i].keys():
                duration = json_data["traceEvents"][i].get("dur")

            if 'ts' in json_data["traceEvents"][i].keys():
                start = json_data["traceEvents"][i].get("ts")

            if 'args' in json_data["traceEvents"][i].keys():
                if 'shape' in json_data["traceEvents"][i]['args'].keys():
                    shape_str = json_data["traceEvents"][i]['args'].get("shape")
                    if bool(shape_str.strip().replace('(', '').replace(')', '')):
                        shape = shape_str
            events_df = events_df.append(
                pd.DataFrame([[name, start, duration, occurences, shape, is_DLS_Op]], columns=data_columns),
                ignore_index=True)

        events_df = events_df.sort_values(by="Duration", ascending=False)
        total = calculate_total_time(events_df)

        dls_opts_df = events_df[events_df["Is_DLS_Op"] == True]
        dls_opts_df = dls_opts_df.sort_values(by="Duration", ascending=False)
        dls_time_sum = dls_opts_df["Duration"].sum()

        var_read_op = variable_read_op_time(events_df)
        print_total_vs_dls_time(total, dls_time_sum, var_read_op)
        print_profiler_results_dls_only(dls_opts_df, total)
        # print_profiler_results_all(events_df, total)
        generate_otput_csv_file(events_df, dls_opts_df, path_to_logs)
