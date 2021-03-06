import bay

# print('Doing sorted estimates...')
# print('-------------------------')
# moorings = bay.read_all_moorings(minimal=True, avoid_wda=False)
# bay.make_merged_nc(moorings, fileprefix='bay_merged_sorted')
# bay.generate_mean_median_dataframe('bay_merged_sorted_hourly.nc', '../estimates/mean_median_KT_sorted.csv')

print('\n\nDoing mooring estimates...')
print('-------------------------')
# moorings = bay.read_all_moorings(minimal=True, avoid_wda=True)
# moorings = [bay.read_nrl3(minimal=True), bay.read_nrl4(minimal=True), bay.read_nrl5(minimal=True)]
# bay.make_merged_nc(moorings, fileprefix='bay_merged_8n')
bay.generate_mean_median_dataframe('bay_merged_8n_hourly.nc', '../estimates/8n_mean_median_KT_mooring.csv')
