import csv
import json
from collections import defaultdict

def sort_by_category(input_fname, output_fname):

	input_fh = open(input_fname, 'r')
	output_fh = open(output_fname, 'w')

	reader = csv.reader(input_fh)

	sorted_data = defaultdict(lambda: [])
	for row in reader:
		features = row[1].split()
		class_name = row[2]
		sorted_data[class_name].append([class_name] + features)

	output_fh.write(json.dumps(sorted_data, indent=2))

	input_fh.close()
	output_fh.close()

	


