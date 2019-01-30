import bay

moorings = bay.read_all_moorings(minimal=True)
bay.make_merged_nc(moorings)
bay.generate_mean_median_dataframe()
