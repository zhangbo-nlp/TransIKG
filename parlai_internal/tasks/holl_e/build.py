#!/usr/bin/env python3

import parlai.core.build_data as build_data
import os
from parlai.core.build_data import DownloadableFile

RESOURCES = [
    DownloadableFile(
        '1XLrXU2_64FBVt3-3UwdprdyAGXOIc8ID',
        'train_data.json',
        'ab16ea1ba3227d8f78e5b76e0a6bc7a06d4155ce2ebc09ff96f614eadb92022a',
        from_google=True,
        zipped=False,
    ),
    DownloadableFile(
        '1hSGhG0HyZSvwU855R4FsnDRqxLursPmi',
        'test_data.json',
        '5bf7fe251fdf7a6ab16ea7a73a5ae54631384f50a6c17eff3b27b16fa4e9fb81',
        from_google=True,
        zipped=False,
    ),
    DownloadableFile(
        '1BIQ8VbXdndRSDaCkPEruaVv_8WegWeok',
        'multi_reference_test.json',
        'e53dac184fba2969ea606b1ec4c8b4297f443551d588ba3156b032e489938188',
        from_google=True,
        zipped=False,
    ),
]


def build(opt):
    dpath = os.path.join(opt['datapath'], 'holl_e')
    version = ''

    if not build_data.built(dpath, version_string=version):
        print('[building data: ' + dpath + ']')
        if build_data.built(dpath):
            # An older version exists, so remove these outdated files.
            build_data.remove_dir(dpath)
        build_data.make_dir(dpath)

        # Download the data.
        for downloadable_file in RESOURCES:
            downloadable_file.download_file(dpath)

        # Mark the data as built.
        build_data.mark_done(dpath, version_string=version)
