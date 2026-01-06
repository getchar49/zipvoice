import re
from roman import fromRoman
#import nltk
#nltk.download('punkt')
from nltk import word_tokenize
# import tts_norm.VietnameseTextNormalizer
# from num2words import num2words
from vietnam_number import n2w, n2w_single
from app.normalizer.cores import *

from app.normalizer.abbre_vn import ALPHABET
from app.normalizer.measure import MEASURE_DICT
from app.normalizer.verbatim import VERBATIM
from app.normalizer.units import UNITS_DICT
class TextNormalizer():
    def __init__(self):
        super().__init__()

    def lowercase(self, input_str):
        return input_str.strip().lower().strip()

    def tokenize(input_str):
        """
        'abc@gmail.com' --> 'abc @ gmail.com'
        """
        tokens = word_tokenize(input_str)
        output_str = " ".join(tokens)

        return output_str

    def remove_multi_space(self, input_str):
        return re.sub(' +', ' ', input_str)

    def separate_numbers_adjacent_chars(self, sentence):
        separated_sentence = re.sub(r'(\d)(?=[^\d.,h:])|(?<=[^\d.,h:])(\d)', r'\1 \2', sentence)
        return separated_sentence

    def replace_special_words(self, input_str):
        dict_map = {
            " AI ": " ây ai ",
            " KIA ": " ki a ",
            " IT ": " ai ti "
        }
        for i, j in dict_map.items():
            input_str = input_str.replace(i, j)
        return input_str

    def remove_special_characters_v1(self, input_str):
        input_str = ' ' + input_str + ' '
#         punct = '! " “ \' ( ) ; [ ] * _ ` { | } ~ … 》 ≧ ≦ –  ‘ ’ · 】 ◇◆ ㅁ • ” `` '' ” ● ︶ ︶ ● † ⬔'.split()
        punct = '" “ \' ( ) [ ] * _ ` { ~ } … 》 ≧ ≦ –  ‘ ’ · 】 ◇◆ ㅁ • ” `` '' ” ● ︶ ︶ ● † ⬔'.split()
        for e in punct:
            e = e.strip()
            input_str = input_str.replace(e, ' ')
        return input_str.strip()
    
    def norm_duplicate_word(self, input_str):
        # input_str = remove_adjacent_word(input_str, "ngày")
        # input_str = remove_adjacent_word(input_str, "tháng")
        # input_str = remove_adjacent_word(input_str, "năm")
        # input_str = remove_adjacent_word(input_str, "giờ")
        # input_str = remove_adjacent_word(input_str, "phút")
        # input_str = remove_adjacent_word(input_str, "giây")
        # input_str = remove_adjacent_word(input_str, ".")
        # input_str = remove_adjacent_word(input_str, "?")
        input_str = self.remove_redundant_words(input_str, 'ngày', 'ngày')
        input_str = self.remove_redundant_words(input_str, 'mùng', 'ngày')
        input_str = self.remove_redundant_words(input_str, 'tháng', 'tháng')
        return input_str.strip()
    
    def remove_redundant_words(self, text, word1, word2):
        text = text.strip()
        new_text = []
        tokens = [i.strip() for i in text.split()]
#         print('tokens: ', tokens)
        if len(tokens) != 0:
            for i in range(len(tokens)-1):
                if tokens[i] == word1 and tokens[i+1] == word2:
                    continue
                else:
                    new_text.append(tokens[i+1])
#         print('tokens after: ', tokens)
        if tokens:
            new_text.insert(0, tokens[0])
        else:
            pass
        
        return ' '.join(new_text)

    def remove_special_characters_v2(self, input_str):
        return re.sub(r'[^\w\s,.?!;-]', ' ', input_str)

    def remove_emoji(self, input_str):
        """
        Remove emoji in text
        """
        emoji_pattern = re.compile("["
                                u"\U0001F600-\U0001F64F"  # emoticons
                                u"\U0001F300-\U0001F5FF"  # symbols & pictographs
                                u"\U0001F680-\U0001F6FF"  # transport & map symbols
                                u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
                                u"\U00002500-\U00002BEF"  # chinese char
                                u"\U00002702-\U000027B0"
                                u"\U00002702-\U000027B0"
                                u"\U000024C2-\U0001F251"
                                u"\U0001f926-\U0001f937"
                                u"\U00010000-\U0010ffff"
                                u"\u2640-\u2642"
                                u"\u2600-\u2B55"
                                u"\u200d"
                                u"\u23cf"
                                u"\u23e9"
                                u"\u231a"
                                u"\ufe0f"  # dingbats
                                u"\u3030"
                                "]+", flags=re.UNICODE)
        return emoji_pattern.sub(r'', input_str)
    
    # def remove_emoticons(self, input_str):
    #     """
    #     Remove emoticons in text
    #     """
    #     emoticon_pattern = re.compile(u'(' + u'|'.join(k for k in EMOTICONS) + u')')
    #     return emoticon_pattern.sub(r'', input_str)

    def remove_urls(self, input_str):
        """
        Remove urls in input_str
        """
        url_pattern = re.compile(r'(https|http)?://\S+|www\.\S+|[A-Za-z0-9]*@[A-Za-z]*\.+[A-Za-z0-9]*')
        return url_pattern.sub(r'', input_str)
    
    def remove_html(self, input_str):
        """
        Remove html in input_str
        """
        html_pattern = re.compile('<.*?>')
        return html_pattern.sub(r'', input_str)
    
    # def norm_vnmese_accent(self, input_str):
    #     output_str = VietnameseTextNormalizer.Normalize(input_str)
    #     return output_str

    def normalize_phone_number(self, input_str):
        """
        Normalize phone numbers strictly based on context.
        """
        if not input_str:
            return ""

        # 1. Define the Prefix (Context)
        # Note: [hH] matches h or H. No pipe needed inside [].
        # We use (?:...) for non-capturing groups to save memory.
        prefix_pattern = r"(?:[hH]otline|[T|t]ổng đài|[Đ|đ]iện thoại|SDT|SĐT|[zZ]alo|đường dây nóng|[Ll]iên hệ|gọi|call|chi tiết|hỗ trợ|tư vấn|liên lạc|công ty|[bB]án [hH]àng|[đĐ]ặt hàng)"

        # 2. Define the Separator (CRITICAL FIX)
        # Only allow spaces, colons, or dots immediately after the prefix. 
        # distinct from the "greedy" (.*) in the original code.
        separator_pattern = r"[\s:.]+"

        # 3. Define the Phone Number
        # Removed \d{3} and \d{4} standalone matches to avoid matching years/distances.
        # Added logic for standard VN mobile/landlines and extensions.
        number_pattern = r"(?:\+?\d{8,12}|\d{3,5}\s\d{3,4}\s\d{3,4}|\d{4}\.\d{3}\.\d{3}|\d{3}\.\d{3}\.\d{4}|\d{3}\.\d{4}\.\d{3}|\d{4}\s\d{2}\s\d{2}\s\d{2})"

        # Combine them: Look for Prefix + Separator + Number
        full_pattern = f"({prefix_pattern})({separator_pattern})({number_pattern})"
        
        # We use a compilation flag IgnoreCase to simplify [hH] handling if desired, 
        # but kept your style for now.
        regex = re.compile(full_pattern)

        # We will build a new string to avoid the 'replace' bug
        result_str = ""
        last_end = 0

        # finditer is safer and faster than the while loop approach
        for match in regex.finditer(input_str):
            # Append text before the match
            result_str += input_str[last_end:match.start()]
            
            # Extract parts
            prefix_found = match.group(1)
            separator_found = match.group(2)
            original_number = match.group(3)

            # Convert the number
            converted_number = phone2words(original_number)

            # Reconstruct the string with the converted number
            # We keep the prefix (e.g., "Hotline: ") and add the converted words
            result_str += f"{prefix_found}{separator_found} {converted_number} "
            
            last_end = match.end()

        # Append the rest of the string
        result_str += input_str[last_end:]

        return result_str.strip()

    def norm_tag_verbatim(self, input_str):
        input_str = ' ' + input_str + ' '
        
        for key, value in VERBATIM.items():
            input_str = input_str.replace(' ' + key.strip() + ' ', ' ' + value.strip() + ' ')
            
        return input_str.strip()

    def normalize_AZ09(self, input_str):
        """
        Normalize sequences with forms [A-Z]{1}[0-9]{1,2} or [0-9]{1,2}[A-Z]{1},
        i.e. 'chung cư A10', 'cục phòng chống tội phạm công nghệ cao C50'
        """
        input_str = ' ' + input_str + ' '
        number_letter = re.findall('([a-zA-Z]{1,10}\d{1,10}[a-zA-Z]{1,10}\d{1,10}|\d{1,10}[a-zA-Z]{1,10}\d{1,10}[a-zA-Z]{1,10}|[a-zA-Z]{1,10}\d{1,10}[a-zA-Z]{1,10}|\d{1,10}[a-zA-Z]{1,10}\d{1,10}|[a-zA-Z]{1,10}\d{1,10}|\d{1,10}[a-zA-Z]{1,10})', input_str)
        if len(number_letter) > 0:
            for item in number_letter:
                item_str = item
                numbers = re.findall(r'\d+', item)
                characters = re.findall(r'[a-zA-Z]+', item_str)

                for character in characters:
                    temp_character_str = ''
                    for i in range(len(character)):
                        temp_character_str += ' ' + ALPHABET[character[i]] + ' '
                    item_str = item_str.replace(character, ' ' + temp_character_str + ' ')

                for number in numbers:
                    # if int(number) < 10:
                    item_str = item_str.replace(number, ' ' + n2w_single(number) + ' ')
                    # else:
                    #     item_str = item_str.replace(number, ' ' + num2words_fixed(number) + ' ')

                input_str = input_str.replace(item,  ' ' + item_str + ' ')
                    
        return input_str.strip()

    def normalize_date(self, input_str):
        """
        Normalize dates.
        """
        input_str = re.sub(r'\s+', ' ', input_str)
  
        input_str = self.norm_date_type_0(input_str)
        #print(f'1: {input_str}')
        input_str = self.norm_date_type_1(input_str)
        #print(f'2: {input_str}')
        input_str = self.norm_date_type_2(input_str)
        #print(f'3: {input_str}')
        input_str = self.norm_date_type_3(input_str)
        #print(f'4: {input_str}')
        input_str = self.norm_date_type_4(input_str)
        #print(f'5: {input_str}')
        input_str = self.norm_date_type_5(input_str)
        #print(f'6: {input_str}')
        # input_str = self.norm_date_type_6(input_str)
        #print(f'7: {input_str}')
        input_str = self.norm_date_type_7(input_str)
        return input_str.strip()


#     def norm_date_type_0(self, input_str):
#         input_str = ' ' + input_str + ' '
#         # Normalize dd/mm/yy[yy] (dmy) form of dates
#         # Note: '8-6-2019' format này để riêng vì tránh cases "từ '8-6/2019' mây thay đổi nhiều"
#         date_dmy_pattern = re.compile(r'([Nn]gày)\s((0?[1-9]|[12]\d|3[01])[\/.](0?[1-9]|[1][0-2])[\/.](\d{4})|(0?[1-9]|[12]\d|3[01])[\-](0?[1-9]|[1][0-2])[\-](\d{4}))([\s|.|,|)|;|:])')
#         temp_str_date_dmy = input_str
#         dates_dmy = []

#         while(date_dmy_pattern.search(temp_str_date_dmy)):
#             date = date_dmy_pattern.search(temp_str_date_dmy)
#             term = date.group()
#             x = re.search(r"(0?[1-9]|[12]\d|3[01])[\/.](0?[1-9]|[1][0-2])[\/.](\d{4})|(0?[1-9]|[12]\d|3[01])[\-](0?[1-9]|[1][0-2])[\-](\d{4})", term)
#             dates_dmy.append(x.group())
#             temp_str_date_dmy = temp_str_date_dmy[date.span()[1]-1:]

#         if len(dates_dmy) > 0:
#             for date in dates_dmy:
#                 date_str = date_dmy2words(date)
#                 # print('date_dd/mm/[yy]yy:', date, '-', input_str)
#                 input_str = input_str.replace(date, ' ' + date_str + ' ')
#         return input_str.strip()
    
    def norm_date_type_0(self, input_str):
        input_str = ' ' + input_str + ' '
        # Normalize dd/mm/yy[yy] (dmy) form of dates
        # Note: '8-6-2019' format này để riêng vì tránh cases "từ '8-6/2019' mây thay đổi nhiều"
        # date_dmy_pattern = re.compile(r'([Nn]gày)\s((0?[1-9]|[12]\d|3[01])[\/.](0?[1-9]|[1][0-2])[\/.](\d{4})|(0?[1-9]|[12]\d|3[01])[\-](0?[1-9]|[1][0-2])[\-](\d{4}))([\s|.|,|)|;|:])')
        date_dmy_pattern = re.compile(r'([Nn]gày)\s((0?[1-9]|[12]\d|3[01])\s*[\/.]\s*(0?[1-9]|[1][0-2])\s*[\/.]\s*(\d{4})|(0?[1-9]|[12]\d|3[01])\s*[\-]\s*(0?[1-9]|[1][0-2])\s*[\-]\s*(\d{4}))([\s|.|,|)|;|:])')        
        temp_str_date_dmy = input_str
        dates_dmy = []

        while(date_dmy_pattern.search(temp_str_date_dmy)):
            date = date_dmy_pattern.search(temp_str_date_dmy)
            term = date.group()
            x = re.search(r"(0?[1-9]|[12]\d|3[01])\s*[\/.]\s*(0?[1-9]|[1][0-2])\s*[\/.]\s*(\d{4})|(0?[1-9]|[12]\d|3[01])\s*[\-]\s*(0?[1-9]|[1][0-2])\s*[\-]\s*(\d{4})", term)
            dates_dmy.append(x.group())
            temp_str_date_dmy = temp_str_date_dmy[date.span()[1]-1:]

        if len(dates_dmy) > 0:
            for date in dates_dmy:
                date_str = date_dmy2words(date)
                # print('date_dd/mm/[yy]yy:', date, '-', input_str)
                input_str = input_str.replace(date, ' ' + date_str + ' ')
        return input_str.strip()
    
    def norm_date_type_1(self, input_str):
        input_str = ' ' + input_str + ' '
        
        # Normalize dd/mm/yy[yy] (dmy) form of dates
        # Note: '8-6-2019' format này để riêng vì tránh cases "từ '8-6/2019' mây thay đổi nhiều"
        #date_dmy_pattern = re.compile(r'[\s|(](0?[1-9]|[12]\d|3[01])[\/.](0?[1-9]|[1][0-2])[\/.](\d{4}|\d{2})|\s(0?[1-9]|[12]\d|3[01])[\-](0?[1-9]|[1][0-2])[\-](\d{4}|\d{2})([\s|.|,|)|:|;])')
        date_dmy_pattern = re.compile(r'(?<!\d)(0?[1-9]|[12]\d|3[01])([\/.\-])(0?[1-9]|1[0-2])\2(\d{2}|\d{4})(?![\d.])')
        temp_str_date_dmy = input_str
        dates_dmy = []
        while(date_dmy_pattern.search(temp_str_date_dmy)):
            date = date_dmy_pattern.search(temp_str_date_dmy)
            dates_dmy.append(date.group())

            temp_str_date_dmy = temp_str_date_dmy[date.span()[1]-1:]
        
        if len(dates_dmy) > 0:
            for date in dates_dmy:
                date_str = date_dmy2words(date)
                # print('date_dd/mm/[yy]yy:', date, '-', input_str)
                input_str = input_str.replace(date, ' ' + date_str + ' ')

        return input_str.strip()

    def norm_date_type_2(self, input_str):
        input_str = ' ' + input_str + ' '
        
        # 1. Define the Pattern
        # Group 1: Prefix (từ, đến, etc.)
        # Group 2: The full Date string (e.g., 21/11)
        # Group 5: The suffix/delimiter (comma, space, dot, etc.)
        pattern_str = r'(sau ,|mai ,|qua ,|nay ,|sớm|đến hết|[đ|Đ]ợt|[Pp]hiên|[Nn]gày|[D|d]ịp|[Ss]áng|[Tt]rưa|[Cc]hiều|[Tt]ối|[Đđ]êm|[Mm]ùng|[Hh]ôm|nay|[Ss]áng qua|[Tt]ưa qua|[Cc]hiều qua|[Tt]ối qua|[Đđ]êm qua|[Hh]ôm qua|[Hh]ôm sau|mai|[Vv]ào|[Kk]éo dài tới|[Dd]ự kiến tới|[Đđ]ến|[Tt]ới|[Tt]ừ)\(*\s*((0?[1-9]|[12]\d|3[01])[\/\-.](0?[1-9]|[1][0-2]))([\s|.|,|)|:|;])'
        date_dm_pattern = re.compile(pattern_str)

        # 2. Define a replacement callback function
        def replace_match(match):
            prefix = match.group(1)      # e.g., "đến"
            date_raw = match.group(2)    # e.g., "21/11"
            delimiter = match.group(5)   # e.g., "," (The crucial part to keep!)
            
            # Convert only the date part using your helper function
            # (Assuming self.date_dm2words exists)
            date_text = date_dm2words(date_raw) 
            
            # Return the reconstructed string: Prefix + space + Converted Date + Delimiter
            return f"{prefix} {date_text}{delimiter}"

        # 3. Apply substitution
        input_str = date_dm_pattern.sub(replace_match, input_str)

        # 4. Handle the Special "và" cases (Keeping your original logic here roughly)
        dates_dm_special = re.findall('[Nn]gày .+ và (\d{1,2}[\/\-.]\d{1,2})\s', input_str)
        for date in dates_dm_special:
            date_str = date_dm2words(date)
            # Use simple replace here, or upgrade to regex sub for safety as well
            input_str = input_str.replace(date, ' ' + date_str + ' ')

        return input_str.strip()

    def norm_date_type_3(self, input_str):
        input_str = ' ' + input_str + ' '
        # Normalize dd/mm form without clear rules
        # Cám ơn cha dành cho Ngày của cha (16/6) tới đây của Hồ Ngọc Hà...
        p = re.compile(r'[Nn]gày.*\s\(*\s*((0?[1-9]|[12]\d|3[01])\s*\/\s*(0?[1-9]|[1][0-2]))([\s|.|,|)|;|:])')
        l = []
        temp_line = input_str
        while(p.search(temp_line)):
            item = p.search(temp_line)
            l.append(item.group())
            temp_line = temp_line[item.span()[1]-1:]

        dates_dm_ = []
        if len(l) > 0:
            temp_str = l[0]
            p = re.compile(r'\s(0?[1-9]|[12]\d|3[01])\s*\/\s*(0?[1-9]|[1][0-2])')
            while(p.search(temp_str)):
                date = p.search(temp_str)
                dates_dm_.append(date.group())
                temp_str = temp_str[date.span()[1]-1:]

        if len(dates_dm_) > 0:
            for date in dates_dm_:
                date_str = date_dm2words(date)
                input_str = input_str.replace(date, ' ' + date_str + ' ')

        return input_str.strip()

    def norm_date_type_4(self, input_str):
        input_str = ' ' + input_str + ' '
        # Normalize mm/yyyy (my) form of dates
        # @improve:
        # những cases không có [Tt]háng ở trước --> thêm 'tháng' ở date_my2word()
        # trong read.py  nhưng tránh các trường hợp Quý 2/2018, đợt 3/2019, tỷ lệ 1/2000
        date_my_pattern = re.compile(r'\s((0?[1-9]|[1][0-2])[\/\-.](\d{4}))([\s|.|,|)|:|;])')
        temp_str_date_my = input_str
        dates_my = []
        while(date_my_pattern.search(temp_str_date_my)):
            date = date_my_pattern.search(temp_str_date_my)
            dates_my.append(date.group().strip())
            temp_str_date_my = temp_str_date_my[date.span()[1]-1:]
        if len(dates_my) > 0:
            for date in dates_my:
                date_str = date_my2words(date)
                # print('date_mm/yyyy:', date, '-', input_str)
                input_str = input_str.replace(date, ' ' + date_str + ' ')

        return input_str.strip()

    def normalize_date_range(self, input_str):
        input_str = self.norm_date_range_type_1(input_str)
        input_str = self.norm_date_range_type_2(input_str)
        input_str = self.norm_date_range_type_3(input_str)
        input_str = self.norm_date_range_type_4(input_str)
        input_str = self.norm_date_range_type_5(input_str)
        input_str = self.norm_date_range_type_6(input_str)

        return input_str.strip()

    def norm_date_range_type_1(self, input_str):
        """
        Normalize date ranges.
        """
        input_str = ' ' + input_str + ' '
        # Normalize yyyy-yyyy forms: 2016-2017, 1912-1982 ngày sinh, v.v.
        # @improve:
        # Khi nào thì chèn từ vào ví dụ 'năm học 2018-2019' thì đọc luôn tên năm còn
        # 'công ty thu thiếu hụt khoản này từ năm 2012-2017 là 11,3 tỷ đồng'
        # thì cần thêm từ "đến": từ năm 2012 đến 2017
        
    def norm_date_range_type_1(self, input_str):
        year_range_pattern = re.compile(r'\s(\d{4})\s*\-\s*(\d{4})([\s|.|,|)])')
        temp_str = input_str
        year_range_list = []

        while(year_range_pattern.search(temp_str)):
            year_range = year_range_pattern.search(temp_str)
            year_range_list.append(year_range.group())
            temp_str = temp_str[year_range.span()[1]-1:]

        if len(year_range_list) > 0:
            # print(str(year_range_list), " : ", input_str)
            for year_range in year_range_list:
                year_range_norm = year_range.replace('-', ' - ')
                year_range_norm = " ".join(year_range_norm.split())
                start_year = year_range_norm.split('-')[0]
                end_year = year_range_norm.split('-')[1]
                year_range_str = ' năm ' + num2words_fixed(start_year) + ' đến ' + ' năm ' + num2words_fixed(end_year)
                input_str = input_str.replace(year_range, ' ' + year_range_str + ' ')

        return input_str.strip()

    def norm_date_range_type_2(self, input_str):
        input_str = ' ' + input_str + ' '
        # Normalize dd/mm/yyyy-dd/mm/yyyy forms
        date_range_dmy_pattern = re.compile(r'\s((0?[1-9]|[12]\d|3[01])[\/.](0?[1-9]|[1][0-2])[\/.](\d{4})\s*\-\s*(0?[1-9]|[12]\d|3[01])[\/.](0?[1-9]|[1][0-2])[\/.](\d{4}))([\s|.|,|)])')
        temp_str = input_str
        date_range_dmy_list = []
        
        while(date_range_dmy_pattern.search(temp_str)):
            date_range_dmy = date_range_dmy_pattern.search(temp_str)
            date_range_dmy_list.append(date_range_dmy.group())
            temp_str = temp_str[date_range_dmy.span()[1]-1:]

        if len(date_range_dmy_list) > 0:
            # print(str(date_range_dmy_list), " : ", input_str)
            for date_range_dmy in date_range_dmy_list:
                start_date = date_range_dmy.split('-')[0]
                end_date = date_range_dmy.split('-')[1]
                date_range_dmy_str = date_dmy2words(start_date) + ' đến ' + date_dmy2words(end_date)
                input_str = input_str.replace(date_range_dmy, ' ' + date_range_dmy_str + ' ')
                
        return input_str.strip()

    def norm_date_range_type_3(self, input_str):
        input_str = ' ' + input_str + ' '
        # Normalize dd-dd/mm/yyyy forms
        date_range_dmy1_pattern = re.compile(r'\s((0?[1-9]|[12]\d|3[01])\s*\-\s*(0?[1-9]|[12]\d|3[01])[\/.](0?[1-9]|[1][0-2])[\/.](\d{4}))([\s|.|,|)])')
        temp_str = input_str
        date_range_dmy1_list = []

        while(date_range_dmy1_pattern.search(temp_str)):
            date_range_dmy1 = date_range_dmy1_pattern.search(temp_str)
            date_range_dmy1_list.append(date_range_dmy1.group())
            temp_str = temp_str[date_range_dmy1.span()[1]-1:]

        if len(date_range_dmy1_list) > 0:
            # print(str(date_range_dmy1_list), " : ", input_str)
            for date_range_dmy1 in date_range_dmy1_list:
                start_date = date_range_dmy1.split('-')[0]
                end_date = date_range_dmy1.split('-')[1]
                date_range_dmy1_str = ' ngày ' + num2words_fixed(start_date) + ' đến ' + date_dmy2words(end_date)
                input_str = input_str.replace(date_range_dmy1, ' ' + date_range_dmy1_str + ' ')

        return input_str.strip()

    def norm_date_range_type_4(self, input_str):
        input_str = ' ' + input_str + ' '
        # Normalize dd/mm-dd/mm/yyyy forms
        date_range_dmy2_pattern = re.compile(r'\s((0?[1-9]|[12]\d|3[01])[\/.](0?[1-9]|[1][0-2])\s*\-\s*(0?[1-9]|[12]\d|3[01])[\/.](0?[1-9]|[1][0-2])[\/.](\d{4}))([\s|.|,|)])')
        temp_str = input_str
        date_range_dmy2_list = []

        while(date_range_dmy2_pattern.search(temp_str)):
            date_range_dmy2 = date_range_dmy2_pattern.search(temp_str)
            date_range_dmy2_list.append(date_range_dmy2.group())
            temp_str = temp_str[date_range_dmy2.span()[1]-1:]

        if len(date_range_dmy2_list) > 0:
            # print(str(date_range_dmy2_list), " : ", input_str)
            for date_range_dmy2 in date_range_dmy2_list:
                start_date = date_range_dmy2.split('-')[0]
                end_date = date_range_dmy2.split('-')[1]
                date_range_dmy2_str = date_dm2words(start_date) + ' đến ' + date_dmy2words(end_date)
                input_str = input_str.replace(date_range_dmy2, ' ' + date_range_dmy2_str + ' ')

        return input_str.strip()

    def norm_date_range_type_5(self, input_str):
        input_str = ' ' + input_str + ' '
        # Normalize dd/mm-dd/mm forms: 20/1-18/2
        date_range_dm1_pattern = re.compile(r'\s((0?[1-9]|[12]\d|3[01])[\/.](0?[1-9]|[1][0-2])\s*\-\s*(0?[1-9]|[12]\d|3[01])[\/.](0?[1-9]|[1][0-2]))([\s|.|,|)])')
        temp_str = input_str
        date_range_dm1_list = []

        while(date_range_dm1_pattern.search(temp_str)):
            date_range_dm1 = date_range_dm1_pattern.search(temp_str)
            date_range_dm1_list.append(date_range_dm1.group())
            temp_str = temp_str[date_range_dm1.span()[1]-1:]

        if len(date_range_dm1_list) > 0:
            # print(str(date_range_dm1_list), " : ", input_str)
            for date_range_dm1 in date_range_dm1_list:
                start_date = date_range_dm1.split('-')[0]
                end_date = date_range_dm1.split('-')[1]
                date_range_dm1_str = date_dm2words(start_date) + ' đến ' + date_dm2words(end_date)
                input_str = input_str.replace(date_range_dm1, ' ' + date_range_dm1_str + ' ')

        return input_str.strip()

    def norm_date_range_type_6(self, input_str):
        input_str = ' ' + input_str + ' '
        # Normalize dd-dd/mm forms: 15-18/6, 15 -18/6, 15- 18/6
        date_range_dm2_pattern = re.compile(r'\s((0?[1-9]|[12]\d|3[01])\s*\-\s*(0?[1-9]|[12]\d|3[01])[\/.](0?[1-9]|[1][0-2]))([\s|.|,|)])')
        temp_str = input_str
        date_range_dm2_list = []

        while(date_range_dm2_pattern.search(temp_str)):
            date_range_dm2 = date_range_dm2_pattern.search(temp_str)
            date_range_dm2_list.append(date_range_dm2.group())
            temp_str = temp_str[date_range_dm2.span()[1]-1:]

        if len(date_range_dm2_list) > 0:
            # print(str(date_range_dm2_list), " : ", input_str)
            for date_range_dm2 in date_range_dm2_list:
                start_date = date_range_dm2.split('-')[0]
                end_date = date_range_dm2.split('-')[1]
                date_range_dm2_str = ' ngày ' + num2words_fixed(start_date) + ' đến ' + date_dm2words(end_date)
                input_str = input_str.replace(date_range_dm2, ' ' + date_range_dm2_str + ' ')

        return input_str.strip()

    def norm_tag_roman_num(self, input_str):
        input_str = self.norm_tag_roman_num_v1(input_str)
        input_str = self.norm_tag_roman_num_v2(input_str)
        return input_str.strip()

    def norm_tag_roman_num_v2(self, input_str):
        """
        Normalize roman numerals to Vietnamese text.
        Example: "Chương IV" -> "Chương bốn"
        """
        
        # 1. Define Context Keywords
        # These are the words that MUST precede a Roman numeral to be valid.
        keywords = (
            r'(thế hệ|số|đại hội|giai đoạn|quý|cấp|quận|kỳ|khóa|quy định|'
            r'vành đai|vùng|thế kỷ|khu vực|khu|đợt|hạng|báo động|tập|lần|'
            r'lần thứ|trung ương|tw|chương)'
        )

        # 2. Define Roman Numeral Pattern
        # Fix 1: Add (?=[XVIxvi]) to ensure the match contains at least one Roman character (not empty).
        # Fix 2: We include both upper and lower case in the pattern, but rely on the 'keywords' context to filter out false positives like "tu vi".
        roman_chars = r'(?=[XVIxvi])(?:X{0,3})(?:IX|IV|V?I{0,3}|ix|iv|v?i{0,3})'

        # 3. Combined Pattern
        # Structure: (Keyword) + (Space) + (Roman Numeral) + (Word Boundary/Punctuation)
        # We use \b at the start to ensure we don't match middle of words.
        pattern = r'\b({})\s+({})(?=[\s\.,\)\/]|$)' .format(keywords, roman_chars)

        def replace_func(match):
            prefix = match.group(1) # The keyword (e.g., "Chương")
            roman = match.group(2)  # The numeral (e.g., "IV")

            # Basic validation to avoid common Vietnamese words if the keyword check fails somehow
            # "vi" is a common false positive. Only convert "vi" if it's strictly Uppercase (VI) 
            # OR if we are very confident in the keyword.
            # However, since we require the keyword prefix in the regex, "tu vi" will not match 
            # because "tu" is not in your keyword list.
            
            try:
                # Convert Roman to Integer
                roman_int = fromRoman(roman.upper()) # Ensure fromRoman handles uppercase
                
                # Convert Integer to Vietnamese Text
                # normalize space for output
                roman_text = num2words_fixed(str(roman_int))
                
                return f"{prefix} {roman_text}"
            except Exception as e:
                # If conversion fails (invalid roman), return original text
                return match.group(0)

        # 4. Perform Substitution
        # flags=re.IGNORECASE allows "chương" or "Chương" to match.
        return re.sub(pattern, replace_func, input_str, flags=re.IGNORECASE)

    def norm_tag_roman_num_v1(self, input_str):
        input_str = ' ' + input_str + ' '
        roman_numeral_p = re.compile('(\s(\(\s*X{0,3})(IX|IV|V?I{0,3})|\s(\(\s*x{0,3})(ix|iv|v?i{0,3}))([\s|.|,|)|/])')
        temp_str = input_str
        # temp_str = " " + " ".join(word_tokenize(temp_str)) + " "
        roman_numeral_list = []

        while (roman_numeral_p.search(temp_str, re.IGNORECASE)):
            roman_numeral = roman_numeral_p.search(temp_str)
            roman_numeral_list.append(roman_numeral.group().strip())
            temp_str = temp_str[roman_numeral.span()[1] - 1:]

        if len(roman_numeral_list) > 0:
            for roman_numeral in roman_numeral_list:
                roman_numeral = roman_numeral.replace(' ', '')
                for character in '. , ( ) /'.split():
                    roman_numeral = roman_numeral.replace(character, '')
                roman = roman_numeral.strip()
                if roman != '':
                    roman2int = fromRoman(roman.upper())
                    roman_numeral_str = num2words_fixed(str(roman2int))
                    input_str = input_str.replace(roman_numeral, ' ' + roman_numeral_str + ' ')

        return input_str.strip()

    def normalize_time(self, input_str):
        input_str = self.normalize_time1(input_str)
        input_str = self.normalize_time2(input_str)
        return input_str.strip()

    def normalize_time2(self, input_str):
        """
        Normalize time
        """
        input_str = ' ' + input_str + ' '
        time_patterns = re.compile(r'(\d+)(\:|h)(0?[0-9]|[1-5][0-9])(\:|p)([1-5][0-9]|0?[0-9])(\s|s)|(\d+)(\:|h)([1-5][0-9]|0?[0-9])|(\d+)h')
        temp_str_time = input_str
        times = []

        while(time_patterns.search(temp_str_time)):
            time = time_patterns.search(temp_str_time)
            times.append(time.group())
            temp_str_time = temp_str_time[time.span()[1]:]

        times = [time for time in times if not(time.startswith('24') and (time[3:]>'00')) ]
        if len(times) > 0:
            for time in times:
                time_str = time2words(time)
                input_str = input_str.replace(time, ' ' + time_str + ' ')
        
        return input_str.strip()

    def normalize_time1(self, input_str):
        """
        Normalize time
        """
        input_str = ' ' + input_str + ' '
        # time_patterns = re.compile(r'\b(0?[0-9]|1\d|2[0-4])[:hg](0?[0-9]|[1-5]\d|)\b')
        time_patterns = re.compile(r'((\d+)(\:|h)(0?[0-9]|[1-5][0-9])(\:|p)([1-5][0-9]|0?[0-9])|(\d+)(\:|h)([1-5][0-9]|0?[0-9])|(\d+)h)\s*-\s*((\d+)(\:|h)(0?[0-9]|[1-5][0-9])(\:|p)([1-5][0-9]|0?[0-9])|(\d+)(\:|h)([1-5][0-9]|0?[0-9])|(\d+)h)')
        temp_str_time = input_str
        duration_times = []

        while(time_patterns.search(temp_str_time)):
            time = time_patterns.search(temp_str_time)
            duration_times.append(time.group())
            temp_str_time = temp_str_time[time.span()[1]:]

        for duration_time in duration_times:
            time_patterns = re.compile(r'(\d+)(\:|h)(0?[0-9]|[1-5][0-9])(\:|p)([1-5][0-9]|0?[0-9])|(\d+)(\:|h)([1-5][0-9]|0?[0-9])|(\d+)h')
            temp_time = duration_time
            times = []

            while(time_patterns.search(temp_time)):
                time = time_patterns.search(temp_time)
                times.append(time.group())
                temp_time = temp_time[time.span()[1]:]
            times = [time for time in times if not(time.startswith('24') and (time[3:]>'00')) ]
            time_str = time2words(times[0]) + ' đến ' + time2words(times[1])
            input_str = input_str.replace(duration_time, ' ' + time_str + ' ')

        return input_str.strip()

    def norm_multiply_number(self, input_str):
        input_str = ' ' + input_str + ' '
        vi_numbers = re.findall(r'[0-9]*[.,]*[0-9]+\s*x\s*[0-9]*[.,]*[0-9]+\s*x\s*[0-9]*[.,]*[0-9]|[0-9]*[.,]*[0-9]+\s*x\s*[0-9]*[.,]*[0-9]+', input_str)

        if len(vi_numbers) > 0:
            for vi_number in vi_numbers:
                vi_number_ver = multiply(vi_number)
                input_str = input_str.replace(vi_number, ' ' + vi_number_ver + ' ')

        return input_str.strip()

    def normalize_sport_score(self, input_str):
        """
        Normalize sport scores
        """
        input_str = ' ' + input_str + ' '
        temp_str = input_str.lower()
        scores = re.findall(r'(\s[0-9]+(\-|\:)[0-9]+)', temp_str)
        scores = [score[0] for score in scores]
        sport_ngrams = ['tỷ số', 'chiến thắng', 'trận đấu', 'tỉ số', 'bàn thắng', 'trên sân', 'đội bóng', 'thi đấu', 'cầu thủ',\
                        'vô địch', 'mùa giải', 'đánh bại', 'đối thủ', 'bóng đá', 'gỡ hòa', 'chung kết', 'bán kết', 'ghi bàn', \
                        'chủ nhà', 'tiền đạo', 'dứt điểm', 'tiền vệ', 'tiền đạo', 'thua', 'bị dẫn']
        is_sport = 0
        for item in sport_ngrams:
            if (temp_str.find(item)) != -1:
                is_sport = 1
                break
        if is_sport == 1 and len(scores) > 0:
            for score in scores:
                if '-' in score:
                    lscore = num2words_fixed(score.split('-')[0])
                    rscore = num2words_fixed(score.split('-')[1])
                    score_norm = lscore + ' ' + rscore
                    input_str = input_str.replace(score, ' ' + score_norm + ' ')
                else:
                    lscore = num2words_fixed(score.split(':')[0])
                    rscore = num2words_fixed(score.split(':')[1])
                    score_norm = lscore + ' ' + rscore
                    input_str = input_str.replace(score, ' ' + score_norm + ' ')

        return input_str.strip()

    def normalize_number_plate(self, input_str):
        input_str = self.normalize_number_plate0(input_str)
        input_str = self.normalize_number_plate1(input_str)
        input_str = self.normalize_number_plate2(input_str)
        input_str = self.normalize_number_plate3(input_str)
        input_str = self.normalize_number_plate4(input_str)
        return input_str.strip()

    def normalize_number_plate1(self, input_str):
        """
            Normalize plate
        """
        input_str = ' ' + input_str + ' '
        p = re.compile(r"([B|b]iển [K|k]iểm [S|s]oát|[B|b]iển [S|s]ố xe|[B|b]iển [S|s]ố)+\s*(.)\s*([0-9]+[a-zA-Z]+[0-9]*)(\s|\-)+[0-9]+\.[0-9]+([\s|.|,|)])")
        temp_str = input_str
        number_plate_list = []
        while(p.search(temp_str)):
            number_plate = p.search(temp_str)
            x = number_plate.group()
            number_plate_list.append(x.strip().split("-")[-1] if '-' in x else x.split()[-1])
            temp_str = temp_str[number_plate.span()[1]-1:]

        if len(number_plate_list) > 0:
            for number_plate in number_plate_list:
                number_plate_str = phone2words(number_plate)
                input_str = input_str.replace(number_plate, ' ' + number_plate_str + ' ')

        return input_str.strip()

    def normalize_number_plate2(self, input_str):
        """
            Normalize plate
        """
        input_str = ' ' + input_str + ' '
        p = re.compile(r"([B|b]iển [K|k]iểm [S|s]oát|[B|b]iển [S|s]ố xe|[B|b]iển [S|s]ố)+\s*(.)\s*([0-9]+[a-zA-Z]+[0-9]*)+\s*(\-|\.|\s)\s*[0-9]+([\s|.|,|)])")
        temp_str = input_str
        number_plate_list = []
        while(p.search(temp_str)):
            number_plate = p.search(temp_str)
            x = number_plate.group()
            if '-' in x:
                number_plate_list.append(x.split("-")[-1])
            elif '.' in x:
                number_plate_list.append(x.split(".")[-1])
            else:
                number_plate_list.append(x.split()[-1])
            temp_str = temp_str[number_plate.span()[1]-1:]

        if len(number_plate_list) > 0:
            for number_plate in number_plate_list:
                number_plate_str = phone2words(number_plate)
                input_str = input_str.replace(number_plate, ' ' + number_plate_str + ' ')

        return input_str.strip()

    def normalize_number_plate3(self, input_str):
        """
            Normalize plate
        """
        input_str = ' ' + input_str + ' '
        p = re.compile(r"([B|b]iển [K|k]iểm [S|s]oát|[B|b]iển [S|s]ố xe|[B|b]iển [S|s]ố)+\s*.*\s*([0-9]+[a-zA-Z]+[0-9]*)(\s|\-)+[0-9]+\.[0-9]+([\s|.|,|)])")
        temp_str = input_str
        number_plate_list = []
        while(p.search(temp_str)):
            number_plate = p.search(temp_str)
            x = number_plate.group()
            number_plate_list.append(x.strip().split("-")[-1] if '-' in x else x.split()[-1])
            temp_str = temp_str[number_plate.span()[1]-1:]

        if len(number_plate_list) > 0:
            for number_plate in number_plate_list:
                number_plate_str = phone2words(number_plate)
                input_str = input_str.replace(number_plate, ' ' + number_plate_str + ' ')

        return input_str.strip()

    def normalize_number_plate4(self, input_str):
        """
            Normalize plate
        """
        input_str = ' ' + input_str + ' '
        p = re.compile(r"([B|b]iển [K|k]iểm [S|s]oát|[B|b]iển [S|s]ố xe|[B|b]iển [S|s]ố)+\s*.*\s*([0-9]+[a-zA-Z]+[0-9]*)+\s*(\-|\.|\s)\s*[0-9]+")
        temp_str = input_str
        number_plate_list = []
        while(p.search(temp_str)):
            number_plate = p.search(temp_str)
            x = number_plate.group()
            if '-' in x:
                number_plate_list.append(x.split("-")[-1])
            elif '.' in x:
                number_plate_list.append(x.split(".")[-1])
            else:
                number_plate_list.append(x.split()[-1])
            temp_str = temp_str[number_plate.span()[1]-1:]
        if len(number_plate_list) > 0:
            for number_plate in number_plate_list:
                number_plate_str = phone2words(number_plate)
                input_str = input_str.replace(number_plate, ' ' + number_plate_str + ' ')

        return input_str.strip()

    def normalize_number_plate0(self, input_str):
        """
            Norm several types of Vietnamese license plates including white, red, motor, car,...
            eg: '29LD-888.99' -> '2 9 L D 8 8 8 9 9'
                '80-NG-167-76' -> '8 0 N G 1 6 7 7 6'
        """
        plate_pattern = r'\b(?:\d{2}[A-Z]{1,2}\d?-?\d{3}\.\d{2}|\d{2}-\d{3}-[A-Z]{2}-\d{2}|[A-Z]{2}-\d{2}-\d{2}|\d{2}-[A-Z]{2}-\d{3}-\d{2})\b'
        matches = re.findall(plate_pattern, input_str)
        if len(matches) > 0:
            for item in matches:
                new_item = item.replace('-', '').replace('.', '')
                new_item = ' '.join(new_item)
                input_str = input_str.replace(item.strip(), new_item.strip())
        return input_str.strip()

    def norm_id_digit(self, input_str):
        """
            Normalize CMT, STK
        """
        input_str = ' ' + input_str + ' '
        # p = re.compile(r"[\d ]{9,20}")
        p = re.compile(r"([C|c]hứng minh nhân dân|[C|c]hứng minh thư|[M|m]ã thẻ|[S|s]ố thẻ|[S|s]ố tài khoản|[C|c]ăn cước|[C|c]ăn cước công dân|[M|m]ã số thuế|[B|b]iển số|[M|m]ã số|nhân viên|mã)+\s+.{1,20}(\:|là\s*)*\s*(\d{2,20}\b)")  
        temp_str = input_str
        digits_list = []
        while(p.search(temp_str)):
            digit = p.search(temp_str)
            term = digit.group()
            x = re.search(r"\d{2,20}\b",term)
            digits_list.append(x.group())
            temp_str = temp_str[digit.span()[1]-1:]
        # digits = re.findall(r"[\d ]{9,20}", input_str)
        if len(digits_list) > 0:
            # print(digits_list)
            for digit in digits_list:
                digits_str = phone2words(digit)
                input_str = input_str.replace(digit, ' ' + digits_str + ' ')

        return input_str.strip()

    def normalize_negative_number(self, input_str):
        input_str = ' ' + input_str + ' '
        p = re.compile(r'\s\-([0-9]*,*[0-9]+)\s')
        # p = re.compile(r"(là|kết quả|âm|dưới|lạnh|xuống|nhiệt độ|áp suất)+\s*\:*\-\s*[0-9]*,*[0-9]+")
        temp_str = input_str
        neg_numbers = []
        while (p.search(temp_str)):
            numbers = p.search(temp_str)
            term = numbers.group()
            neg_numbers.append(term.split("-")[-1])
            temp_str = temp_str[numbers.span()[1] - 1:]
        neg_numbers = [number.replace(',', '.') for number in neg_numbers]
        neg_numbers.sort(key=float)
        neg_numbers = [number.replace('.', ',') for number in neg_numbers]

        if len(neg_numbers) > 0:
            for number in neg_numbers:
                if ',' in number:
                    numbers_str = num2words_float(number)
                else:
                    numbers_str = num2words_fixed(number)
                numbers_str = ' âm ' + numbers_str
                input_str = input_str.replace("-" + number, ' ' + numbers_str + ' ')

        return input_str.strip()


    def norm_soccer(self, input_str):
        # Normalize units of VFF football team: U23, U19, etc
        matches = re.findall('\sU[\-\.]*[0-9][0-9][\s|.|,|)]', input_str)
        if len(matches) > 0:
            for item in matches:
                item_norm = item.replace('.','').replace('-','').replace(' U', ' U ')
                input_str = input_str.replace(item, item_norm)

        return input_str.strip()

    def norm_abbre(self, input_str, abbre_dict):
        temp_str = input_str

        # not remove punctuation
        words_temp = temp_str.split()
        for i in range(len(words_temp)):
            temp_word = words_temp[i]
            if temp_word[-1] in [',', '.', ')', '}', ']', '!', '?', '/', '-', ':', ';']:
                temp_word = temp_word[:-1]

            if temp_word in abbre_dict.keys():
                input_str =  input_str.replace(temp_word, str(abbre_dict[temp_word]).strip().lower())

        #remove punctuation
        temp_str = input_str
        for character in [',', '.', ')', '}', ']', '!', '?', '/', '-']:
            temp_str = temp_str.replace(character, ' ')

        words_inp = temp_str.split()
        for i in range(len(words_inp)):
            temp_word = words_inp[i]

            if temp_word in abbre_dict.keys():
                input_str =  input_str.replace(temp_word, str(abbre_dict[temp_word]).strip())

        return input_str.strip()

    def norm_tag_fraction(self, input_str):
        input_str = self.norm_tag_fraction1(input_str)
        input_str = self.norm_tag_fraction2(input_str)
        input_str = self.norm_tag_fraction3(input_str)
        input_str = self.norm_tag_fraction4(input_str)
        return input_str.strip()


    def norm_tag_fraction1(self, input_str):
        """
        Normalize number range.
        ###([0-9]+,[0-9]+\s*\/\s*[0-9]+,[0-9]+|[0-9]+,[0-9]+\s*\/\s*[0-9]+|[0-9]+\s*\/\s*[0-9]+,[0-9]+) regex for 2,1/4
        """
        input_str = ' ' + input_str + ' '
        p = re.compile(r"(thứ|hơn|gần|:|hạng|được|tới|góp|là|có|còn|lên|bằng|[Cc]hiếm|giảm|tỷ lệ|tỉ lệ|[K|k]hoảng|online)\s[0-9]+\s*\/\s*[0-9]+[\s|.|,|)|;]")
        temp_str = input_str
        ratio_list = []
        while(p.search(temp_str)):
            ratio = p.search(temp_str)
            x = ratio.group().replace(' / ', '/').replace(' /', '/').replace('/ ', '/')
            ratio_list.append(x.split()[-1])
            temp_str = temp_str[ratio.span()[1]-1:]

        for character in '. , ( ) ;'.split():
            ratio_list = [item.replace(character, '') for item in ratio_list]
        # ratio_list = [item for item in ratio_list if int(item.split('/')[0]) <= int(item.split('/')[1])]
        if len(ratio_list) > 0:
            for ratio in ratio_list:
                first_num = ratio.split('/')[0]
                second_num = ratio.split('/')[1]
                # if int(second_num) > 10:
                #     ratio_str = num2words_fixed(first_num) + ' trên ' + num2words_fixed(second_num)
                # else:
                ratio_str = num2words_fixed(first_num) + ' phần ' + num2words_fixed(second_num)

                input_str = input_str.replace(ratio, ' ' + ratio_str + ' ')

        return input_str.strip()

    def norm_tag_fraction2(self, input_str):
        """
        Normalize case 1/3 muỗng
        """
        input_str = ' ' + input_str + ' '
        p = re.compile(r"\s[0-9]+\s*\/\s*[0-9]+\s*(muỗng|thìa|ly|cốc|chén|chai|lọ)")
        temp_str = input_str
        ratio_list = []
        while(p.search(temp_str)):
            ratio = p.search(temp_str)
            x = ratio.group().replace(' / ', '/').replace(' /', '/').replace('/ ', '/')
            ratio_list.append(x.split()[0])
            temp_str = temp_str[ratio.span()[1]-1:]

        for character in '. , ( ) ;'.split():
            ratio_list = [item.replace(character, '') for item in ratio_list]
        # ratio_list = [item for item in ratio_list if int(item.split('/')[0]) <= int(item.split('/')[1])]
        ratio_list = sorted(ratio_list, key=len, reverse=True)
        if len(ratio_list) > 0:
            for ratio in ratio_list:
                first_num = ratio.split('/')[0]
                second_num = ratio.split('/')[1]
                ratio_str = num2words_fixed(first_num) + ' phần ' + num2words_fixed(second_num)
                input_str = input_str.replace(ratio, ' ' + ratio_str + ' ')

        return input_str.strip()

    def norm_tag_fraction3(self, input_str):
        """
        Normalize case Điều 48 Nghị định 110/2013/NĐ-CP
        """
        input_str = ' ' + input_str + ' '
        p = re.compile(r"([N|n]ghị [Đ|đ]ịnh|[N|n]ghị [Q|q]uyết|[T|t]hông [T|t]ư|[T|t]hông [T|t]ư liên tịch)\s*[0-9]+\s*\/\s*[0-9]+(\/)*[0-9]+")
        temp_str = input_str
        ratio_list = []
        while(p.search(temp_str)):
            ratio = p.search(temp_str)
            x = ratio.group().replace(' / ', '/').replace(' /', '/').replace('/ ', '/')
            ratio_list.append(x.split()[-1])
            temp_str = temp_str[ratio.span()[1]-1:]

        for character in '. , ( ) ;'.split():
            ratio_list = [item.replace(character, '') for item in ratio_list]
        ratio_list = sorted(ratio_list, key=len, reverse=True)
        # ratio_list = [item for item in ratio_list if int(item.split('/')[0]) <= int(item.split('/')[1])]
        if len(ratio_list) > 0:
            for ratio in ratio_list:
                ratio_str = ratio.replace('/', ', năm ')
                input_str = input_str.replace(ratio, ' ' + ratio_str + ' ')
        return input_str.strip()

    def norm_tag_fraction4(self, input_str):
        """
        Normalize case trường hợp/100.000 dân
        """
        input_str = ' ' + input_str + ' '
        p = re.compile(r"\s[AĂÂÁẮẤÀẰẦẢẲẨÃẴẪẠẶẬĐEÊÉẾÈỀẺỂẼỄẸỆIÍÌỈĨỊOÔƠÓỐỚÒỒỜỎỔỞÕỖỠỌỘỢUƯÚỨÙỪỦỬŨỮỤỰYÝỲỶỸỴA-Zaăâáắấàằầảẳẩãẵẫạặậđeêéếèềẻểẽễẹệiíìỉĩịoôơóốớòồờỏổởõỗỡọộợuưúứùừủửũữụựyýỳỷỹỵa-z]+\/")
        temp_str = input_str
        ratio_list = []
        while(p.search(temp_str)):
            ratio = p.search(temp_str)
            x = ratio.group().replace(' / ', '/').replace(' /', '/').replace('/ ', '/')
            ratio_list.append(x.split()[-1])
            temp_str = temp_str[ratio.span()[1]-1:]

        # ratio_list = [item for item in ratio_list if int(item.split('/')[0]) <= int(item.split('/')[1])]
        if len(ratio_list) > 0:
            for ratio in ratio_list:
                ratio_str = ratio.replace('/', ' trên ')
                input_str = input_str.replace(ratio, ' ' + ratio_str + ' ')
        return input_str.strip()


    def norm_adress(self, input_str):
        """
        Normalize case ngõ 12/124 
        """
        input_str = ' ' + input_str + ' '
        p = re.compile(r"([N|n]gõ|[N|n]gách|[H|h]ẻm)\s[0-9]+\s*\/\s*[0-9]+(\/)*[0-9]*[\s|.|,|)|;]")
        temp_str = input_str
        ratio_list = []
        while(p.search(temp_str)):
            ratio = p.search(temp_str)
            x = ratio.group().replace(' / ', '/').replace(' /', '/').replace('/ ', '/')
            ratio_list.append(x.split()[-1])
            temp_str = temp_str[ratio.span()[1]-1:]

        for character in '. , ( ) ;'.split():
            ratio_list = [item.replace(character, '') for item in ratio_list]
        ratio_list = sorted(ratio_list, key=len, reverse=True)
        # ratio_list = [item for item in ratio_list if int(item.split('/')[0]) <= int(item.split('/')[1])]
        if len(ratio_list) > 0:
            for ratio in ratio_list:
                ratio_str = ratio.replace('/', ' trên ')
                input_str = input_str.replace(ratio, ' ' + ratio_str + ' ')
        return input_str.strip()

    def normalize_number(self, input_str):
        input_str = self.norm_number_type_0(input_str)
        input_str = self.norm_number_type_1(input_str)
        input_str = self.norm_number_type_2(input_str)
        input_str = self.norm_number_type_3(input_str)

        return input_str.strip()

    def norm_number_type_0(self, input_str):
        """
        Normalize number
        """
        # Normalize vi-style numbers: '2.300 Euro', '25.320 vé', etc
        input_str = ' ' + input_str + ' '
        vi_numbers = re.findall(r'\b\d{1,3}(?:\.\d{3})*,\d+\b', input_str)
        # print(vi_numbers)
        if len(vi_numbers) > 0:
            for vi_number in vi_numbers:
                vi_number_1 = vi_number.split(',')[0]
                vi_number_2 = vi_number.split(',')[1]
                vi_number_1 = "".join(vi_number_1.split('.'))
                vi_number_1_str = num2words_fixed(vi_number_1)
                if int(vi_number_2) == 0:
                    vi_number_ver = vi_number_1_str
                else:
                    vi_number_2_str = num2words_fixed(vi_number_2)
                    vi_number_ver = vi_number_1_str + " phảy " + vi_number_2_str
                input_str = input_str.replace(vi_number, ' ' + vi_number_ver + ' ')

        return input_str.strip()

    def norm_number_type_1(self, input_str):
        """
        Normalize number
        """
        # Normalize vi-style numbers: '2.300 Euro', '25.320 vé', etc
        input_str = ' ' + input_str + ' '
        vi_numbers = re.findall(r'[\s|(]([\d]+\.[\d]+\.*[\d]*\.*[\d]*\.*[\d]*)', input_str)
        # print(vi_numbers)
        if len(vi_numbers) > 0:
            for vi_number in vi_numbers:
                vi_number_norm = "".join(vi_number.split('.'))
                if int(vi_number_norm) >= 1000:
                    vi_number_str = num2words_fixed(vi_number_norm)
                    input_str = input_str.replace(vi_number, ' ' + vi_number_str + ' ')
                else:
                    vi_number_ver = version2words(vi_number)
                    input_str = input_str.replace(vi_number, ' ' + vi_number_ver + ' ')

        return input_str.strip()

    # def norm_number_type_2(self, input_str):
    #     # Normalize numbers with comma format: '224,3 tỷ', '16,2 phần trăm', etc
    #     input_str = ' ' + input_str + ' '
    #     numbers_w_comma = re.findall(r'[\s|(]([\d]+,[\d]+)', input_str)
    #     if len(numbers_w_comma) > 0:
    #         for number in numbers_w_comma:
    #             number_str = num2words_float(number)
    #             input_str = input_str.replace(number,' ' + number_str + ' ')

    #     return input_str.strip()

    def norm_number_type_2(self, input_str):
        """
        Normalize number
        """
        # Normalize vi-style numbers: '2.300 Euro', '25.320 vé', etc
        input_str = ' ' + input_str + ' '
        vi_numbers = re.findall(r'[\s|(]([\d]+\,[\d]+\,*[\d]*\,*[\d]*\,*[\d]*)', input_str)
        if len(vi_numbers) > 0:
            for vi_number in vi_numbers:
                if vi_number.count(',') == 1 and int(vi_number.replace(',', '')) % 1000 != 0:
                    number_str = num2words_float(vi_number)
                    input_str = input_str.replace(vi_number,' ' + number_str + ' ')
                else:
                    vi_number_norm = "".join(vi_number.split(','))
                    vi_number_str = num2words_fixed(vi_number_norm)
                    input_str = input_str.replace(vi_number, ' ' + vi_number_str + ' ')

        return input_str.strip()

    def norm_number_type_3(self, input_str):
        input_str = ' ' + input_str + ' '
        numbers = re.findall(r'([\d]+)', input_str)
        numbers = sorted([int(number) for number in numbers], reverse=True)
        numbers = [str(number) for number in numbers]
        if len(numbers) > 0:
            for number in numbers:
                
                number_str = num2words_fixed(number)
                input_str = input_str.replace(number,' ' + number_str + ' ')

        return input_str.strip()

    def norm_tag_measure(self, input_str):
        """
        Normalize unit names and number + unit names. i.e.
        'kg' --> 'ki lô gam', '1000mAh' --> 'một nghìn mi li am pe'
        """
        # Normalize unit names (length, area, volume, information, speed, etc)
        input_str = ' ' + input_str + ' '
        input_str = unit2words(input_str)        
        # print(input_str)
        
        measure_pattern = '\s*[0-9]*\.*\,*\-*[0-9]+\s*%MEASURE%'
        # '\s*[0-9]*\.*\,*\-*[0-9]+\s*%MEASURE%(\s+|.)'
        for term, norm_term in MEASURE_DICT.items():
            pattern = measure_pattern.replace("%MEASURE%", term.strip())
            # term = ' ' + term.strip() + ' '
            input_str = self.norm_tag_measure_generic(
                input_str=input_str, 
                pattern=pattern, 
                term=term, 
                norm_term = ' ' + norm_term.strip() + ' ')
        return input_str.strip()

    def norm_tag_measure_generic(self, input_str, pattern, term, norm_term):
        matches = re.findall(pattern, input_str)
        # print(len(matches))
        if len(matches) > 0:
            for item in matches:
                # print(item)
                item_norm_out = item.replace(term, norm_term)
                input_str = input_str.replace(' ' + item.strip() + ' ', ' ' + item_norm_out.strip() + ' ')
        return input_str.strip()


    def normalize_number_range(self, input_str):
        input_str = self.normalize_number_range1(input_str)
        input_str = self.normalize_number_range2(input_str)
        return input_str.strip()

    def normalize_number_range1(self, input_str):
        """
        Normalize number ranges
        """
        input_str = ' ' + input_str + ' '
        p = re.compile(r"(hơn|kém|gấp|tăng|tầm|giảm|liệu trình|nhất|tới|có|sau|mức|tuổi|từ|tăng tốc|được|khoảng|trong|vòng|dao động|cấp|tốc độ)(.*)\s+[0-9]*(,|\.)*[0-9]+\s*\-\s*[0-9]*(,|\.)*[0-9]+[\s|.|,|)]")
        temp_str = input_str
        number_range_list = []
        while(p.search(temp_str)):
            number_range = p.search(temp_str)
            term = number_range.group()
            x = re.search(r"[0-9]*(,|\.)*[0-9]+\s*\-\s*[0-9]*(,|\.)*[0-9]+[\s|.|,|)]", term)
            number_range_list.append(x.group())
            temp_str = temp_str[number_range.span()[1]-1:]

        if len(number_range_list) > 0:
            for number_range in number_range_list:
                if number_range[-1] in [',', '.', ')']:
                    number_range = number_range[:-1]
                start_num = number_range.split('-')[0]
                end_num = number_range.split('-')[1]
                if ',' in start_num:
                    start_num = num2words_float(start_num)
                elif '.' in start_num:
                    start_num = "".join(start_num.split('.'))
                    if int(start_num) >= 1000:
                        start_num = num2words_fixed(start_num)
                    else:
                        start_num = version2words(start_num)
                else:
                    start_num = num2words_fixed(start_num)

                if ',' in end_num:
                    end_num = num2words_float(end_num)
                elif '.' in end_num:
                    end_num = "".join(end_num.split('.'))
                    if int(end_num) >= 1000:
                        end_num = num2words_fixed(end_num)
                    else:
                        end_num = version2words(end_num)
                else:
                    end_num = num2words_fixed(end_num)

                number_range_str = start_num + ' đến ' + end_num
                input_str = input_str.replace(number_range, ' ' + number_range_str + ' ')

        return input_str.strip()

    def normalize_number_range2(self, input_str):
        """
        Normalize number ranges
        """
        input_str = ' ' + input_str + ' '
        p = re.compile(r"\s+[0-9]*,*[0-9]+\s*\-\s*[0-9]*,*[0-9]+\s*(lần|cái|khách|túi|ki lô gam|kg|hôm|ngày|muỗng|thìa|phút|gói|cái|tháng|năm|tiếng|mét|ca|tuổi|phần trăm|ki lô gam|giờ|giây|xen ti mét|mi li mét|độ|lít|tấn|thùng|cái|con|triệu|gam|hàng|m|ki lô mét|h|phòng)")
        temp_str = input_str
        number_range_list = []
        while(p.search(temp_str)):
            number_range = p.search(temp_str)
            term = number_range.group()
            x = re.search(r"[0-9]*,*[0-9]+\s*\-\s*[0-9]*,*[0-9]+[\s|.|,|)]", term)
            number_range_list.append(x.group())
            temp_str = temp_str[number_range.span()[1]-1:]

        
        if len(number_range_list) > 0:
            for number_range in number_range_list:
                start_num = number_range.split('-')[0]
                end_num = number_range.split('-')[1]
                if ',' in start_num:
                    start_num = num2words_float(start_num)
                else:
                    start_num = num2words_fixed(start_num)

                if ',' in end_num:
                    end_num = num2words_float(end_num)
                else:
                    end_num = num2words_fixed(end_num)

                number_range_str = start_num + ' đến ' + end_num
                input_str = input_str.replace(number_range, ' ' + number_range_str + ' ')

        return input_str.strip()

    def normalize_rate(self, input_str):
        """
        Normalize number ranges
        """
        input_str = ' ' + input_str + ' '
        p = re.compile(r"(đánh giá|rate)\s+[0-9]+[*|⭐|★].")
        temp_str = input_str
        rate_list = []
        while(p.search(temp_str)):
            rate = p.search(temp_str)
            term = rate.group()
            x = re.search(r"[*|⭐|★]", term)
            rate_list.append(x.group())
            temp_str = temp_str[rate.span()[1]-1:]
        
        if len(rate_list) > 0:
            for rate in rate_list:
                input_str = input_str.replace(rate, ' ' + 'sao' + ' ')

        return input_str.strip()

    def norm_math_characters(self, input_str):
        """
            Normalize math characters
        """
        input_str = ' ' + input_str + ' '
        input_str = replace_math_characters(input_str)

        return input_str.strip()
    
    def separate_comma_and_dot_at_the_end(self, input_str):
        """
            Separate comma and dot in sentence but keep number in og
            eg: 'tỉ lệ 80%, 90%.' --> 'tỉ lệ 80% , 90% . '
                '10.000.000' --> '10.000.000'
        """
        s = input_str.replace('\r\n', '\n').replace('\r', '\n')
        s = re.sub(r'(?<=[\.\,\?\!\:\;\)\]\}\'"\u2019\u201D])\s*\n+\s*', ' ', s)
        input_str = re.sub(r'\s*\n+\s*', '. ', s)
        input_str = re.sub(r'[:]+', '. ', input_str)
        input_str = re.sub(r'phẩy', 'phảy', input_str)
        separated_sentence = re.sub(r'(?<!\d)([.,])(?!\d)', r'\1 ', input_str)
        # Loại bỏ các khoảng trắng thừa có thể xảy ra do dấu cách giữa các dấu câu
        separated_sentence = re.sub(r'\s+', ' ', separated_sentence).strip()
        return separated_sentence
    
    def norm_ratio(self, input_str):
        """
            Norm ratio
            eg: 'Tỉ lệ 1:18' --> 'tỉ lệ 1 18'
        """
        # ratio_pattern = r"([T|t]ỉ lệ|[T|t]ỷ lệ)+\s*\b\d+:\d+\b"
        matches = []
        ratio_pattern = r"(\b\d+:\d+\b)"
        if "tỉ lệ" in input_str.lower():
            matches = re.findall(ratio_pattern, input_str)
        if len(matches) > 0:
            # print(matches)
            for item in matches:
                new_item = item.replace(':', ' ')
                input_str = input_str.replace(item.strip(), new_item.strip())
        return input_str.strip()    
    
    def norm_date_type_5(self, input_str):
        input_str = ' ' + input_str + ' '
        # match datetime like dd/mm which doesnt have prefix "ngày" but "sáng", "trưa", "chiều", "tối" include space character
        p = re.compile(r'([Ss]áng|[Tt]rưa|[Cc]hiều|[Tt]ối|[Ll]ễ|[Tt]ết|[Hh]ôm|[Đđ]ợt)\s*\(*\s*((0?[1-9]|[12]\d|3[01])\s*\/\s*(0?[1-9]|[1][0-2]))([\s|.|,|)|;|:])')
        l = []
        temp_line = input_str
        while(p.search(temp_line)):
            item = p.search(temp_line)
            l.append(item.group())
            temp_line = temp_line[item.span()[1]-1:]

        dates_dm_ = []
        if len(l) > 0:
            temp_str = l[0]
            p = re.compile(r'\s(0?[1-9]|[12]\d|3[01])\s*\/\s*(0?[1-9]|[1][0-2])')
            while(p.search(temp_str)):
                date = p.search(temp_str)
                dates_dm_.append(date.group())
                temp_str = temp_str[date.span()[1]-1:]

        if len(dates_dm_) > 0:
            for date in dates_dm_:
                date_str = date_dm2words(date)
                input_str = input_str.replace(date, ' ' + date_str + ' ')

        return input_str.strip()
    
    def norm_date_type_6(self, input_str):
        input_str = ' ' + input_str + ' '
        # match case dd/mm like "ngày 31/08 và 1/9"
        p = re.compile(r'(0?[1-9]|[12]\d|3[01])\s*\/\s*(0?[1-9]|1[0-2])\s+(và|đến|)\s+(0?[1-9]|[12]\d|3[01])\s*\/\s*(0?[1-9]|1[0-2])')
        l = []
        temp_line = input_str
        while(p.search(temp_line)):
            item = p.search(temp_line)
            l.append(item.group())
            temp_line = temp_line[item.span()[1]-1:]

        dates_dm_ = []
        if len(l) > 0:
            temp_str = l[0]
            p = re.compile(r'(\s)?(0?[1-9]|[12]\d|3[01])\s*\/\s*(0?[1-9]|[1][0-2])')
            while(p.search(temp_str)):
                date = p.search(temp_str)
                dates_dm_.append(date.group())
                temp_str = temp_str[date.span()[1]-1:]

        if len(dates_dm_) > 0:
            for date in dates_dm_:
                date_str = date_dm2words(date)
                input_str = input_str.replace(date, ' ' + date_str + ' ')

        return input_str.strip()
    
    def normalize_time_range(self, input_str):
        # match time range like 9-10h, 9h - 10 h, not include minute and second
        time_patterns = re.compile(r'\d+\s*h?\s*-\s*\d+\s*h')
        temp_str_time = input_str
        times = []
        while(time_patterns.search(temp_str_time)):
            time = time_patterns.search(temp_str_time)
            times.append(time.group())
            temp_str_time = temp_str_time[time.span()[1]:]
        for matched_time in times:
            time_list = matched_time.split("-")
            if "h" in time_list[0]:
                hour, minute = time_list[0].split("h")
                convert_t1 = num2words_fixed(str(int(hour))) + ' giờ '
            else:
                convert_t1 = num2words_fixed(str(int(time_list[0])))
            if "h" in time_list[1]:
                hour, minute = time_list[1].split("h")
                convert_t2 = num2words_fixed(str(int(hour))) + ' giờ '
            else:
                convert_t2 = num2words_fixed(str(int(time_list[1])))

            input_str = input_str.replace(matched_time," " + convert_t1 + " đến " + convert_t2 + " ")
        return input_str
    
    def norm_unit(self, input_str):
        out = input_str
        MULTIPLIERS = r'(?:chục|trăm|nghìn|ngàn|triệu|tỷ)'
        PUNCT_CLASS = r'(?:\s|[.,;:)\]\}!?\-"\'…]|$)'

        for term, norm_term in UNITS_DICT.items():
            term_esc = re.escape(term)  
            pattern = rf'(?P<num>[+-]?\d+(?:[.,]\d+)*)\s*(?P<mult>{MULTIPLIERS}\s*)?{term_esc}(?={PUNCT_CLASS})'
            prog = re.compile(pattern, re.IGNORECASE)

            def repl(m):
                num = m.group('num')
                mult = (m.group('mult') or '').strip()
                parts = [num]
                if mult:
                    parts.append(mult)
                parts.append(norm_term)
                return ' '.join(parts)

            out = prog.sub(repl, out)

        out = re.sub(r'\s{2,}', ' ', out).strip()
        return out
    """
    def norm_unit(self,input_str):
        measure_pattern = '\s*[0-9]*\.*\,*\-*[0-9]+\s*%MEASURE%[ ,)]|\s*[0-9]*\.*\,*\-*[0-9]+\s*%MEASURE%$'
        for term, norm_term in UNITS_DICT.items():
            pattern = measure_pattern.replace("%MEASURE%", term)
            # print("pattern:",pattern)
            matches = re.findall(pattern, input_str)
            # print("matches:",matches)
            if len(matches) > 0:
                for item in matches:
                    item_norm_out = item.replace(term, " " + norm_term + " ")
                    # print("item_norm_out:",item_norm_out)
                    input_str = input_str.replace(item,item_norm_out)
                    # print("input_str:",input_str)
        return input_str
    """
    def norm_date_type_7(self, input_str):
        input_str = ' ' + input_str + ' '
        #match certain time-related words followed by any character and then followed by a date in the format dd/mm (day/month)
        p = re.compile(r'([Ss]áng|[Tt]rưa|[Cc]hiều|[Tt]ối|[Ll]ễ|[Tt]ết|[Hh]ôm|[Đđ]ợt).+?((0?[1-9]|[12]\d|3[01])\s*/\s*(0?[1-9]|1[0-2]))')
        l = []
        temp_line = input_str
        while(p.search(temp_line)):
            item = p.search(temp_line)
            l.append(item.group())
            temp_line = temp_line[item.span()[1]-1:]

        dates_dm_ = []
        if len(l) > 0:
            temp_str = l[0]
            p = re.compile(r'\s(0?[1-9]|[12]\d|3[01])\s*\/\s*(0?[1-9]|[1][0-2])')
            while(p.search(temp_str)):
                date = p.search(temp_str)
                dates_dm_.append(date.group())
                temp_str = temp_str[date.span()[1]-1:]

        if len(dates_dm_) > 0:
            for date in dates_dm_:
                date_str = date_dm2words(date)
                input_str = input_str.replace(date, ' ' + date_str + ' ')

        return input_str.strip()
    
    def replace_dash_range(self, input_str):
        dash_chars = r"-–—-"
        pattern = rf"(\d)\s?([{dash_chars}])\s?(\d)"
        return re.sub(pattern, r"\1 đến \3", input_str)