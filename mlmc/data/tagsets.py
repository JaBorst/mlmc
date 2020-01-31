"""A Collection of frequently used tagsets for reproducability"""

NER = ["O"] + [y for x in  ["PER", "ORG", "MISC", "LOC"] for y in ["B-"+x,"I-"+x,"E-"+x,"S-"+x]]

POS = ['KOKOM', '$.', 'APPR', 'VMFIN', 'VAPP', 'PRF', 'PPOSAT', 'PIDAT', 'PRELAT', 'PTKANT', 'APPO', 'VVPP',
           'VVIMP', 'NE', 'ADV', 'CARD', 'PWAT', 'PPER', 'VVINF', 'NN', 'ADJD', 'PIAT', 'PTKVZ', 'VVFIN', 'PTKA',
           'PWAV', 'XY', 'ADJA', 'VAIMP', 'VMPP', '-X-', 'KOUS', 'VAINF', 'ITJ', 'PRELS', 'PDAT', 'APZR', 'PIS',
           'FM', '$,', 'PTKZU',  'APPRART', 'PAV', 'VAFIN', '$(', 'PDS', 'PPOSS', 'ART', 'PTKNEG', 'VVIZU', 'KON',
           'TRUNC', 'VMINF','KOUI','PWS']

POS_EN =['VBN', 'PRP', 'SYM', 'JJ', 'RB', 'VBP', 'IN', '.', 'TO', 'RP', ',', 'POS', 'CD', 'PDT', 'RBR', 'VBZ', 'VBG',
         'RBS', 'UH', 'LS', 'VB', 'MD', 'NN|SYM','WDT', '$', 'DT', 'JJS', '-X-', ')', 'NNPS', 'NNS', 'NN', 'EX',
         'NNP', '"', '(', 'WRB', 'WP', 'WP$', "''", 'VBD', ':', 'FW', 'CC', 'JJR', 'PRP$']

GERMEVAL2014 = ["O"] + [z for x in  ["PER", "ORG", "OTH", "LOC"] for y in ["B-"+x,"I-"+x,"E-"+x,"S-"+x] for z in  [y, y+"part", y+"deriv"]]