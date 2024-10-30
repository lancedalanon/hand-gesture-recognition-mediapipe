# -*- coding: utf-8 -*-
# Generated by the protocol buffer compiler.  DO NOT EDIT!
# source: mediapipe/modules/holistic_landmark/calculators/roi_tracking_calculator.proto
"""Generated protocol buffer code."""
from google.protobuf import descriptor as _descriptor
from google.protobuf import descriptor_pool as _descriptor_pool
from google.protobuf import symbol_database as _symbol_database
from google.protobuf.internal import builder as _builder
# @@protoc_insertion_point(imports)

_sym_db = _symbol_database.Default()


from mediapipe.framework import calculator_pb2 as mediapipe_dot_framework_dot_calculator__pb2
try:
  mediapipe_dot_framework_dot_calculator__options__pb2 = mediapipe_dot_framework_dot_calculator__pb2.mediapipe_dot_framework_dot_calculator__options__pb2
except AttributeError:
  mediapipe_dot_framework_dot_calculator__options__pb2 = mediapipe_dot_framework_dot_calculator__pb2.mediapipe.framework.calculator_options_pb2


DESCRIPTOR = _descriptor_pool.Default().AddSerializedFile(b'\nMmediapipe/modules/holistic_landmark/calculators/roi_tracking_calculator.proto\x12\tmediapipe\x1a$mediapipe/framework/calculator.proto\"\xbe\x04\n\x1cRoiTrackingCalculatorOptions\x12Q\n\x10iou_requirements\x18\x01 \x01(\x0b\x32\x37.mediapipe.RoiTrackingCalculatorOptions.IouRequirements\x12S\n\x11rect_requirements\x18\x02 \x01(\x0b\x32\x38.mediapipe.RoiTrackingCalculatorOptions.RectRequirements\x12]\n\x16landmarks_requirements\x18\x03 \x01(\x0b\x32=.mediapipe.RoiTrackingCalculatorOptions.LandmarksRequirements\x1a\'\n\x0fIouRequirements\x12\x14\n\x07min_iou\x18\x01 \x01(\x02:\x03\x30.5\x1a^\n\x10RectRequirements\x12\x1c\n\x10rotation_degrees\x18\x01 \x01(\x02:\x02\x31\x30\x12\x18\n\x0btranslation\x18\x02 \x01(\x02:\x03\x30.1\x12\x12\n\x05scale\x18\x03 \x01(\x02:\x03\x30.1\x1a\x36\n\x15LandmarksRequirements\x12\x1d\n\x12recrop_rect_margin\x18\x01 \x01(\x02:\x01\x30\x32V\n\x03\x65xt\x12\x1c.mediapipe.CalculatorOptions\x18\x86\xa3\xad\x9d\x01 \x01(\x0b\x32\'.mediapipe.RoiTrackingCalculatorOptions')

_globals = globals()
_builder.BuildMessageAndEnumDescriptors(DESCRIPTOR, _globals)
_builder.BuildTopDescriptorsAndMessages(DESCRIPTOR, 'mediapipe.modules.holistic_landmark.calculators.roi_tracking_calculator_pb2', _globals)
if _descriptor._USE_C_DESCRIPTORS == False:
  mediapipe_dot_framework_dot_calculator__options__pb2.CalculatorOptions.RegisterExtension(_ROITRACKINGCALCULATOROPTIONS.extensions_by_name['ext'])

  DESCRIPTOR._options = None
  _globals['_ROITRACKINGCALCULATOROPTIONS']._serialized_start=131
  _globals['_ROITRACKINGCALCULATOROPTIONS']._serialized_end=705
  _globals['_ROITRACKINGCALCULATOROPTIONS_IOUREQUIREMENTS']._serialized_start=426
  _globals['_ROITRACKINGCALCULATOROPTIONS_IOUREQUIREMENTS']._serialized_end=465
  _globals['_ROITRACKINGCALCULATOROPTIONS_RECTREQUIREMENTS']._serialized_start=467
  _globals['_ROITRACKINGCALCULATOROPTIONS_RECTREQUIREMENTS']._serialized_end=561
  _globals['_ROITRACKINGCALCULATOROPTIONS_LANDMARKSREQUIREMENTS']._serialized_start=563
  _globals['_ROITRACKINGCALCULATOROPTIONS_LANDMARKSREQUIREMENTS']._serialized_end=617
# @@protoc_insertion_point(module_scope)
