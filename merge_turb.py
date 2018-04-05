import bay

print('Reading all moorings...')
moorings = bay.read_all_moorings()

print('Merging all moorings...')
bay.make_merged_nc(moorings)
