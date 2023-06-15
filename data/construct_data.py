# read each line
import xml.etree.ElementTree as ET
# take the cpe and name, tokenize for each overlap, assign tag, and shape
from urllib.parse import quote
import re
import json
import csv
import urllib.parse
import random
from random import randrange
import itertools
# tags = """
# 1  B-vendor, I-vendor
# 2  B-product, I-product
# 3  B-version
# 4  B-subversion
# 5  B-platform, I-platform (look for lowercase for and Edition)
# """

def build_tag(stext, key):
    structure = {1: ['I-VENDOR', 'B-VENDOR'],
                 2: ['I-PRODUCT', 'B-PRODUCT'],
                 3: ['B-VERSION'],
                 4: ['B-SUBVERSION'],
                 5: ['I-PLATFORM', 'B-PLATFORM']}

    stext = re.sub(r'\b\S+\b', structure[key][0], stext)
    # contains a sentence sub structure
    if key not in range(3, 4):
        stext = re.sub(r'\b\S+\b', structure[key][1], stext, count=1)
    #print(f'return = {stext}')
    return stext


def annotate_line(text, vendor, product, version, sub_version, platform,recur = False):

    vendor = vendor.replace('_', " ")
    product = product.replace('_', " ")
    tag_text = urllib.parse.unquote(text).lower()

    if re.search(f'{vendor}(?:\s|$)', tag_text):
        tag_text = re.sub(f'{vendor}(?:\s|$)', lambda m: build_tag(m.group(), 1), tag_text, count=1)
    else:
       # print(f'{tag_text} failed to find {vendor}')
        return None

    if re.search(f'{product}(?:\s|$)', tag_text):
        tag_text = re.sub(f'{product}(?:\s|$)', lambda m: build_tag(m.group(), 2), tag_text, count=1)
    else:
        # potential case where a product has the same name, which case we sub it back
        if re.search(f'{product}(?:\s|$)', text.lower()) and not recur:
            return annotate_line(vendor + " " + text, vendor, product, version, sub_version, platform, True)
        else:
            #print(f'{tag_text} failed to find {product}')
            return None

    if re.search(f'{version}(?:\s|$)', tag_text):
        tag_text = re.sub(f'{version}(?:\s|$)', lambda m: build_tag(m.group(), 3), tag_text, count=1)
    else:
       # print(f'{tag_text} failed to find {version}')
        return None

    if sub_version:
        if re.search(f'{sub_version}(?:\s|$)', tag_text):
            tag_text = re.sub(f'{sub_version}(?:\s|$)', lambda m: build_tag(m.group(), 4), tag_text, count=1)
        else:
           # print(f'{tag_text} failed to find {sub_version}')
            return None

    if platform:
        platform = platform.replace('_', " ")
        if re.search(f'{platform}(?:\s|$)', tag_text):
            tag_text = re.sub(f'(for )?{platform}( edition)?', lambda m: build_tag(m.group(), 5), tag_text, count=1)
        else:
            #print(f'{tag_text} failed to find {platform}')
            return None

    test_tag = re.sub(r'(B-VENDOR|I-VENDOR|B-PRODUCT|I-PRODUCT|B-VERSION|I-VERSION|B-SUBVERSION|I-PLATFORM|B-PLATFORM)','',tag_text).strip()
    if len(test_tag) > 0:
        print(test_tag)
        return None


   # if any([x in tag_text for x in urllib.parse.unquote(text).lower().split()]):
       # return None

    return {'words': text,
            'tags': tag_text}


def write_data(name, data_type, subset):
    with open(f'{name}.{data_type}.txt', mode='w') as textfile:
        textfile.writelines([x for x in itertools.chain.from_iterable(itertools.zip_longest([i[data_type] for i in subset],['\n']*len(subset))) if x])


def build_cpe():
    max = 0
    totalcount = 0
    invalidcount = 0

    xml_data = open('official-cpe-dictionary_v2.3.xml', 'r').read()  # Read file
    root = ET.XML(xml_data)  # Parse XML
    passlist = []


    with open('../cpe_broken.csv', mode='w') as cpe_file:
        cpe_writer = csv.writer(cpe_file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
        for i, child in enumerate(root[1:]):
            # if max >= 100:
            # break
            max += 1
            text = child.attrib['name']
            title = None
            for sub in child:
                if len(sub.text) > 0:
                    title = sub.text
                    break
            try:
                seg = text.split(":")
                cpe_type = seg[1] if seg[1:] else None
                vendor = seg[2] if seg[2:] else None
                product = seg[3] if seg[3:] else None
                version = seg[4] if seg[4:] else None
                sub_version = seg[5] if seg[5:] else None
                platform = seg[6].replace('~', "") if seg[6:] else None

                if not sub_version:
                    totalcount +=1
                    dic = annotate_line(title, vendor, product, version, sub_version, platform)
                    if dic is not None:
                        passlist.append(dic)
                    else:
                        invalidcount+=1

            except IndexError:
                print(text)

        random.shuffle(passlist)
        train_set = list(passlist)
        test_set = list()
        test_size = len(passlist) * 0.3

        while len(test_set) < test_size:
            random_index = randrange(len(train_set))
            test_set.append(train_set.pop(random_index))

        test_a = test_set[:len(test_set) // 2]
        test_b = test_set[len(test_set) // 2:]

        write_data('testa', 'words',test_a)
        write_data('testa', 'tags', test_a)

        write_data('testb', 'words', test_b)
        write_data('testb', 'tags', test_b)

        write_data('train', 'words', train_set)
        write_data('train', 'tags', train_set)

        print(len(passlist))
        print(f"Stats\n invalid={invalidcount}, total = {totalcount}")
        # memo to self, finish tonight, just burn anything with the extra field.
        # move to put everything in a json
        # basic classification tonight?

        # break cpe into series of patterns by underscore, attempt to mark start and ends.
        # for anything with a special, look for 'for' or Edition. 

        # continuous pattern matches, regex var?


build_cpe()
