

def get_file_list(benchmark):
    if benchmark == 'spair':
        root = 'SPair-71k/'
        return root, [
            'JPEGImages',
            'Layout',
            'PairAnnotation/trn/pair_annotations.json',
            'PairAnnotation/val/pair_annotations.json',
            'PairAnnotation/test/pair_annotations.json',
        ]