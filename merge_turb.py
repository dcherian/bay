import bay

moorings = bay.read_all_moorings()
bay.make_merged_nc(moorings)
