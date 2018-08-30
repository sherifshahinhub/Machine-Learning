import csv, json
#if you are not using utf-8 files, remove the next line
#sys.setdefaultencoding("UTF-8") #set the encode to utf8
#check if you pass the input file and output file
fileInput = 'News_Category_Dataset.json'
fileOutput = 'output.csv'
inputFile = open(fileInput) #open json file

employee_parsed = json.loads(inputFile)

emp_data = employee_parsed['employee_details']

# open a file for writing

employ_data = open(fileOutput, 'w')

# create the csv writer object

csvwriter = csv.writer(employ_data)

count = 0

for emp in emp_data:

      if count == 0:

             header = emp.keys()

             csvwriter.writerow(header)

             count += 1

      csvwriter.writerow(emp.values())

employ_data.close()
inputFile.close() #close the input file
