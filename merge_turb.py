import bay

print('Doing sorted estimates...')
print('-------------------------')
moorings = bay.read_all_moorings(minimal=True, avoid_wda=False)
bay.make_merged_nc(moorings, fileprefix='bay_merged_sorted')
# bay.generate_mean_median_dataframe('bay_merged_sorted_hourly.nc')

print('\n\nDoing mooring estimates...')
print('-------------------------')
moorings = bay.read_all_moorings(minimal=True, avoid_wda=True)
bay.make_merged_nc(moorings, fileprefix='bay_merged_mooring')
# bay.generate_mean_median_dataframe('bay_merged_sorted_mooring.nc')
