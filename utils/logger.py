def write_to_file(log_file, to_write):
    if log_file is not None:
        with open(log_file, "a") as af:
            af.write(to_write)