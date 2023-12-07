#!/usr/bin/env python3
# -*- coding: utf-8 -*-

'''
	Written by Jiawen, September 2023.
'''
import numpy as np
from pyspark.sql.types import StringType, IntegerType
from petastorm.codecs import ScalarCodec, CompressedImageCodec, NdarrayCodec
from petastorm.unischema import Unischema, UnischemaField

ShapeSchema = Unischema(
    "ShapeSchema",
    [
        UnischemaField("noun_id", np.string_, (), ScalarCodec(StringType()), False),
        UnischemaField("signal_names", np.string_, (None,), NdarrayCodec(), False),
        UnischemaField("signal", np.float32, (None, None), NdarrayCodec(), False),
        UnischemaField("target", np.string_, (), ScalarCodec(StringType()), False),
        UnischemaField("signal_length", np.int_, (), ScalarCodec(IntegerType()), False),
    ],
)