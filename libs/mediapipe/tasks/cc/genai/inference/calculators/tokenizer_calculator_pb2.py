# -*- coding: utf-8 -*-
# Generated by the protocol buffer compiler.  DO NOT EDIT!
# source: mediapipe/tasks/cc/genai/inference/calculators/tokenizer_calculator.proto
"""Generated protocol buffer code."""
from google.protobuf import descriptor as _descriptor
from google.protobuf import descriptor_pool as _descriptor_pool
from google.protobuf import symbol_database as _symbol_database
from google.protobuf.internal import builder as _builder
# @@protoc_insertion_point(imports)

_sym_db = _symbol_database.Default()




DESCRIPTOR = _descriptor_pool.Default().AddSerializedFile(b'\nImediapipe/tasks/cc/genai/inference/calculators/tokenizer_calculator.proto\x12\x10odml.infra.proto\"\x8d\x02\n\x1aTokenizerCalculatorOptions\x12\x12\n\nmax_tokens\x18\x01 \x01(\x05\x12\x18\n\x0espm_model_file\x18\x02 \x01(\tH\x00\x12Y\n\x11tflite_model_file\x18\x04 \x01(\x0b\x32<.odml.infra.proto.TokenizerCalculatorOptions.TfLiteModelFileH\x00\x12\x16\n\x0estart_token_id\x18\x03 \x01(\x05\x1a:\n\x0fTfLiteModelFile\x12!\n\x19spm_model_key_in_metadata\x18\x02 \x01(\tJ\x04\x08\x01\x10\x02\x42\x0c\n\nmodel_fileJ\x04\x08\x05\x10\x06\x42>\n\x1b\x63om.google.odml.infra.protoB\x1fTokenizerCalculatorOptionsProtob\x06proto3')

_globals = globals()
_builder.BuildMessageAndEnumDescriptors(DESCRIPTOR, _globals)
_builder.BuildTopDescriptorsAndMessages(DESCRIPTOR, 'mediapipe.tasks.cc.genai.inference.calculators.tokenizer_calculator_pb2', _globals)
if _descriptor._USE_C_DESCRIPTORS == False:

  DESCRIPTOR._options = None
  DESCRIPTOR._serialized_options = b'\n\033com.google.odml.infra.protoB\037TokenizerCalculatorOptionsProto'
  _globals['_TOKENIZERCALCULATOROPTIONS']._serialized_start=96
  _globals['_TOKENIZERCALCULATOROPTIONS']._serialized_end=365
  _globals['_TOKENIZERCALCULATOROPTIONS_TFLITEMODELFILE']._serialized_start=287
  _globals['_TOKENIZERCALCULATOROPTIONS_TFLITEMODELFILE']._serialized_end=345
# @@protoc_insertion_point(module_scope)