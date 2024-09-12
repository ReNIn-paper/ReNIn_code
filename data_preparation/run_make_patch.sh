#!/bin/bash
# python make_patch.py | tee make_patch.log
python 1_crop_top_bottom.py | tee 1_crop_top_bottom.log
python 2_get_alignment_info.py | tee 2_get_alignment_info.log
python 3_align_images.py | tee 3_align_images.log
python 4_make_patch.py | tee 4_make_patch.log
echo "Data preparation done."