import xlrd
from datetime import date,datetime


def get_results(file, result, name, index):
    ExcelFile = xlrd.open_workbook(file)
    table = ExcelFile.sheet_by_index(index)
    for i in range(table.nrows):
        row = table.row_values(i)
        author = row[0]
        strs = row[1]
        strs = strs.split('\n')
        error_list = []
        for j in range(len(strs)):
            str_tmp = strs[j]
            str_tmp = str_tmp.split('(')[-1].strip()
            if str_tmp == ")" or str_tmp == "-)":
                error_list.append(str(j + 1))
        if len(error_list) > 0:
            if author in result:
                result[author][name] = error_list
                result[author][name+'index'] = str(i + 1)
            else:
                result[author] = {name: error_list}
                result[author][name + 'index'] = str(i + 1)
    return result


result = {}
result = get_results('paper.xlsx', result, 'edit', 0)
result = get_results('paper.xlsx', result, 'pro', 1)

keys = list(result.keys())
keys.sort()

f = open('result.txt', 'w')
for t in range(len(keys)):
    key = keys[t]
    # for key in keys:
    value = result[key]
    print(value)
    f.write(str(t + 1)+ ' ' + key + ' ')
    if 'edit' in value:
        f.write(' (' + value['editindex'] + ') ')
        strs = ' '.join(value['edit'])
        f.write(strs)
    f.write(' | ')
    if 'pro' in value:
        f.write(' (' + value['proindex'] + ') ')
        strs = ' '.join(value['pro'])
        f.write(strs)
    f.write('\n')
f.close()






