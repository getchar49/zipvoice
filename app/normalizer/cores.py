from vietnam_number import n2w, n2w_single
# from num2words import num2words
from app.normalizer.units import UNITS_DICT

# def unit2words(input_text):
#     for key, value in UNITS_DICT.items():
#         input_text = input_text.replace(key, f' {value} ')
        
#     return input_text
def unit2words(input_text):
    for key, value in UNITS_DICT.items():
        # key = " " + key.strip() + " "
        input_text = input_text.replace(key, f' {value} ')
    return input_text

def date_dmy2words(date):
    # print(date)
    if date[-1] in '! " “ \' ( ) , . : ; ? [ ] _ ` { | } ~ … — 》 ‘ ’'.split():
        date = date[:-1]
    if '/' in date:
        day, month, year = date.split("/") 
    elif '.' in date:
        day, month, year = date.split(".") 
    elif '-' in date:
        day, month, year = date.split("-") 

    date_str = ' ngày ' + num2words_fixed(str(int(day))) + ' tháng ' + num2words_fixed(str(int(month))) + ' năm ' +  num2words_fixed(year)
    return date_str

def replace_math_characters(input_str):
    input_str = input_str.replace('²', ' bình phương ')
    input_str = input_str.replace('π', ' pi ')
    return input_str

def date_dm2words(date):
    if date[-1] in '! " “ \' ( ) , . : ; ? [ ] _ ` { | } ~ … — 》 ‘ ’'.split():
        date = date[:-1]
    if '/' in date:
        day, month = date.split("/")
    elif '-' in date:
        day, month = date.split("-")
    else:
        day, month = date.split(".")
    date_str = ' ngày ' + num2words_fixed(str(int(day))) + ' tháng ' + num2words_fixed(str(int(month)))
    return date_str

def date_my2words(date):
    if date[-1] in '! " “ \' ( ) , . : ; ? [ ] _ ` { | } ~ … — 》 ‘ ’'.split():
        date = date[:-1]

    if '/' in date:
        month, year = date.split("/") 
    elif '.' in date:
        month, year = date.split(".") 
    elif '-' in date:
        month, year = date.split("-") 

    date_str = ' tháng ' + num2words_fixed(str(int(month))) + ' năm ' +  num2words_fixed(year)
    return date_str

def num2words_fixed(num):
    if num[-1] in '! " “ \' ( ) , . : ; ? [ ] _ ` { | } ~ … — 》 ‘ ’'.split():
        num = num[:-1]

    if int(num) < 1000000:
        if int(num) == 0:
            num_str = n2w(str(int(num)))
        else:
            num_str = n2w(num).replace("lẽ", "lẻ")
            if int(num) % 1000 == 0:
                num_str = n2w(num).replace("không trăm", "")

    elif int(num) >=  1000000 and int(num) < 1000000000:
        if int(num) % 1000 == 0:

            num_str = n2w(num).replace("lẽ", "lẻ").replace('không trăm nghìn', '')[:-10]
        elif int(num) % 1000 != 0:
            print(num)
            print(n2w(num))
            num_str = n2w(num).replace("lẽ", "lẻ").replace('không trăm nghìn', '')
        else:
            num_str = n2w(num).replace("lẽ", "lẻ")

    elif int(num) >=  1000000000 and int(num) < 1000000000000:
        if int(num) % 1000 == 0:
            num_str = n2w(num).replace("lẽ", "lẻ").replace('không trăm triệu', '').replace('không trăm nghìn', '')[:-10]
        elif int(num) % 1000 != 0:
            num_str = n2w(num).replace("lẽ", "lẻ").replace('không trăm triệu', '').replace('không trăm nghìn', '')
        else:
            num_str = n2w(num).replace("lẽ", "lẻ")
    return num_str

def time2words(time):
    if time[-1] == 's':
        time = time[:-1]
    if ":" in time:
        time_split = time.split(":")
        if len(time_split) == 2:
            hour, minute = time_split
            time_str = num2words_fixed(str(int(hour))) + ' giờ ' +  num2words_fixed(str(int(minute))) + " phút "

        elif len(time_split) == 3:
            hour, minute, second = time_split
            time_str = num2words_fixed(str(int(hour))) + ' giờ ' +  num2words_fixed(str(int(minute))) + " phút " + num2words_fixed(str(int(second))) + " giây "
    elif "h" in time:
        hour, minute = time.split("h")
        if minute != "":
            if 'p' in minute:
                minute, second = minute.split('p')
                time_str = num2words_fixed(str(int(hour))) + ' giờ ' +  num2words_fixed(str(int(minute))) + " phút " + num2words_fixed(str(int(second))) + " giây "
            else:
                time_str = num2words_fixed(str(int(hour))) + ' giờ ' +  num2words_fixed(str(int(minute))) + " phút "
        else:
            time_str = num2words_fixed(str(int(hour))) + ' giờ '
    time_str = time_str.replace('không phút', '')
    return time_str

def multiply(input_str):
    element_split = input_str.split("x")
    multiply_str_list = []
    for element in element_split:
        multiply_str_list.append(' ' + n2w(str(element)) + ' ')
    multiply_str = ' nhân '.join(multiply_str_list)
    return multiply_str

def phone2words(number):
    # print(number)
    if number[-1] in '! " “ \' ( ) , . : ; ? [ ] _ ` { | } ~ … — 》 ‘ ’'.split():
        number = number[:-1]
    number = number.replace(' ', '')
    return " " + n2w_single(number) + " "

def num2words_float(number):
    if number[-1] in '! " “ \' ( ) , . : ; ? [ ] _ ` { | } ~ … — 》 ‘ ’'.split():
        number = number[:-1]
    interger, decimal = number.split(',')
    return num2words_fixed(interger) + ' phẩy ' + num2words_fixed(decimal)

def version2words(number):
    if number[-1] in '! " “ \' ( ) , . : ; ? [ ] _ ` { | } ~ … — 》 ‘ ’'.split():
        number = number[:-1]
    interger, decimal = number.split('.')
    return num2words_fixed(interger) + ' chấm ' + phone2words(decimal)