import xlrd
import xlwt

def deal_fn(file, index, name):
    fw = xlwt.Workbook()
    sheet1 = fw.add_sheet(name, cell_overwrite_ok=True)
    excel = xlrd.open_workbook(file)
    table = excel.sheet_by_index(index)
    row_index = 0
    for i in range (table.nrows):
        row = table.row_values(i)
        author = row[0]
        sheet1.write(row_index, 0, author)
        values = row[2]
        values = values.split('\n')
        for j in range(len(values)):
            value = values[j]
            value = value.split(')')
            if len(value) != 4:
                continue
            chair = value[0].split('(')[-1].strip()
            conf  =value[1].split('(')[-1].strip()
            years = value[2].split('(')[-1].strip()
            has_years = ''
            if years == '' or years == '-':
                years = ''
                has_years = 1
            else:
                if len(years) == 4:
                    conf = conf + ', ' + years
                else:
                    years_arr = years.split('-')
                    if len(years_arr) > 1:
                        conf = conf + ', ' + years_arr[0]
                    else:
                        years_arr = years.split(',')
                        if len(years_arr) > 1:
                            years_arr.sort()
                            conf = conf + ', ' + years_arr[0]
                        else:
                            print(i + 1, j + 1)
            sheet1.write(row_index, 1, conf)
            sheet1.write(row_index, 2, chair)
            sheet1.write(row_index, 3, years)
            sheet1.write(row_index, 4, has_years)
            row_index += 1
    fw.save('result.xls')


if __name__ == '__main__':
    deal_fn('main.xlsx', 5, 'edit')
