³'
Ç##
B
AddV2
x"T
y"T
z"T"
Ttype:
2	
B
AssignVariableOp
resource
value"dtype"
dtypetype
~
BiasAdd

value"T	
bias"T
output"T" 
Ttype:
2	"-
data_formatstringNHWC:
NHWCNCHW
N
Cast	
x"SrcT	
y"DstT"
SrcTtype"
DstTtype"
Truncatebool( 
h
ConcatV2
values"T*N
axis"Tidx
output"T"
Nint(0"	
Ttype"
Tidxtype0:
2	
8
Const
output"dtype"
valuetensor"
dtypetype
,
Exp
x"T
y"T"
Ttype:

2
W

ExpandDims

input"T
dim"Tdim
output"T"	
Ttype"
Tdimtype0:
2	
^
Fill
dims"
index_type

value"T
output"T"	
Ttype"

index_typetype0:
2	
­
GatherV2
params"Tparams
indices"Tindices
axis"Taxis
output"Tparams"

batch_dimsint "
Tparamstype"
Tindicestype:
2	"
Taxistype:
2	
.
Identity

input"T
output"T"	
Ttype
q
MatMul
a"T
b"T
product"T"
transpose_abool( "
transpose_bbool( "
Ttype:

2	

Max

input"T
reduction_indices"Tidx
output"T"
	keep_dimsbool( " 
Ttype:
2	"
Tidxtype0:
2	
e
MergeV2Checkpoints
checkpoint_prefixes
destination_prefix"
delete_old_dirsbool(
=
Mul
x"T
y"T
z"T"
Ttype:
2	

NoOp
U
NotEqual
x"T
y"T
z
"	
Ttype"$
incompatible_shape_errorbool(
M
Pack
values"T*N
output"T"
Nint(0"	
Ttype"
axisint 
C
Placeholder
output"dtype"
dtypetype"
shapeshape:

Prod

input"T
reduction_indices"Tidx
output"T"
	keep_dimsbool( " 
Ttype:
2	"
Tidxtype0:
2	
@
ReadVariableOp
resource
value"dtype"
dtypetype
>
RealDiv
x"T
y"T
z"T"
Ttype:
2	
E
Relu
features"T
activations"T"
Ttype:
2	
[
Reshape
tensor"T
shape"Tshape
output"T"	
Ttype"
Tshapetype0:
2	
¥
ResourceGather
resource
indices"Tindices
output"dtype"

batch_dimsint "
validate_indicesbool("
dtypetype"
Tindicestype:
2	
o
	RestoreV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0
l
SaveV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0
?
Select
	condition

t"T
e"T
output"T"	
Ttype
P
Shape

input"T
output"out_type"	
Ttype"
out_typetype0:
2	
H
ShardedFilename
basename	
shard

num_shards
filename
0
Sigmoid
x"T
y"T"
Ttype:

2
[
Split
	split_dim

value"T
output"T*	num_split"
	num_splitint(0"	
Ttype
¾
StatefulPartitionedCall
args2Tin
output2Tout"
Tin
list(type)("
Tout
list(type)("	
ffunc"
configstring "
config_protostring "
executor_typestring 
@
StaticRegexFullMatch	
input

output
"
patternstring
ö
StridedSlice

input"T
begin"Index
end"Index
strides"Index
output"T"	
Ttype"
Indextype:
2	"

begin_maskint "
end_maskint "
ellipsis_maskint "
new_axis_maskint "
shrink_axis_maskint 
N

StringJoin
inputs*N

output"
Nint(0"
	separatorstring 
;
Sub
x"T
y"T
z"T"
Ttype:
2	

Sum

input"T
reduction_indices"Tidx
output"T"
	keep_dimsbool( " 
Ttype:
2	"
Tidxtype0:
2	
-
Tanh
x"T
y"T"
Ttype:

2

TensorListFromTensor
tensor"element_dtype
element_shape"
shape_type
output_handle"
element_dtypetype"

shape_typetype:
2	

TensorListReserve
element_shape"
shape_type
num_elements

handle"
element_dtypetype"

shape_typetype:
2	

TensorListStack
input_handle
element_shape
tensor"element_dtype"
element_dtypetype" 
num_elementsintÿÿÿÿÿÿÿÿÿ
P
	Transpose
x"T
perm"Tperm
y"T"	
Ttype"
Tpermtype0:
2	

VarHandleOp
resource"
	containerstring "
shared_namestring "
dtypetype"
shapeshape"#
allowed_deviceslist(string)
 

While

input2T
output2T"
T
list(type)("
condfunc"
bodyfunc" 
output_shapeslist(shape)
 "
parallel_iterationsint

&
	ZerosLike
x"T
y"T"	
Ttype"serve*2.4.12v2.4.0-49-g85c8b2a817f8÷%

embedding_1/embeddingsVarHandleOp*
_output_shapes
: *
dtype0*
shape:	7*'
shared_nameembedding_1/embeddings

*embedding_1/embeddings/Read/ReadVariableOpReadVariableOpembedding_1/embeddings*
_output_shapes
:	7*
dtype0
u
dense/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	7*
shared_namedense/kernel
n
 dense/kernel/Read/ReadVariableOpReadVariableOpdense/kernel*
_output_shapes
:	7*
dtype0
l

dense/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:7*
shared_name
dense/bias
e
dense/bias/Read/ReadVariableOpReadVariableOp
dense/bias*
_output_shapes
:7*
dtype0

lstm_1/lstm_cell_1/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:
**
shared_namelstm_1/lstm_cell_1/kernel

-lstm_1/lstm_cell_1/kernel/Read/ReadVariableOpReadVariableOplstm_1/lstm_cell_1/kernel* 
_output_shapes
:
*
dtype0
¤
#lstm_1/lstm_cell_1/recurrent_kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:
*4
shared_name%#lstm_1/lstm_cell_1/recurrent_kernel

7lstm_1/lstm_cell_1/recurrent_kernel/Read/ReadVariableOpReadVariableOp#lstm_1/lstm_cell_1/recurrent_kernel* 
_output_shapes
:
*
dtype0

lstm_1/lstm_cell_1/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*(
shared_namelstm_1/lstm_cell_1/bias

+lstm_1/lstm_cell_1/bias/Read/ReadVariableOpReadVariableOplstm_1/lstm_cell_1/bias*
_output_shapes	
:*
dtype0

NoOpNoOp
æ
ConstConst"/device:CPU:0*
_output_shapes
: *
dtype0*¡
valueB B

layer-0
layer_with_weights-0
layer-1
layer-2
layer-3
layer-4
layer-5
layer_with_weights-1
layer-6
layer_with_weights-2
layer-7
	regularization_losses

	variables
trainable_variables
	keras_api

signatures
 
b

embeddings
regularization_losses
	variables
trainable_variables
	keras_api
R
regularization_losses
	variables
trainable_variables
	keras_api
R
regularization_losses
	variables
trainable_variables
	keras_api
 
 
l
cell

state_spec
regularization_losses
	variables
trainable_variables
 	keras_api
h

!kernel
"bias
#regularization_losses
$	variables
%trainable_variables
&	keras_api
 
*
0
'1
(2
)3
!4
"5
*
0
'1
(2
)3
!4
"5
­
*layer_regularization_losses
+non_trainable_variables
	regularization_losses
,metrics

	variables
trainable_variables

-layers
.layer_metrics
 
fd
VARIABLE_VALUEembedding_1/embeddings:layer_with_weights-0/embeddings/.ATTRIBUTES/VARIABLE_VALUE
 

0

0
­
/layer_regularization_losses
0non_trainable_variables
regularization_losses
1metrics
	variables
trainable_variables

2layers
3layer_metrics
 
 
 
­
4layer_regularization_losses
5non_trainable_variables
regularization_losses
6metrics
	variables
trainable_variables

7layers
8layer_metrics
 
 
 
­
9layer_regularization_losses
:non_trainable_variables
regularization_losses
;metrics
	variables
trainable_variables

<layers
=layer_metrics
~

'kernel
(recurrent_kernel
)bias
>regularization_losses
?	variables
@trainable_variables
A	keras_api
 
 

'0
(1
)2

'0
(1
)2
¹
Blayer_regularization_losses
Cnon_trainable_variables
regularization_losses
Dmetrics
	variables
trainable_variables

Estates

Flayers
Glayer_metrics
XV
VARIABLE_VALUEdense/kernel6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUE
TR
VARIABLE_VALUE
dense/bias4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUE
 

!0
"1

!0
"1
­
Hlayer_regularization_losses
Inon_trainable_variables
#regularization_losses
Jmetrics
$	variables
%trainable_variables

Klayers
Llayer_metrics
US
VARIABLE_VALUElstm_1/lstm_cell_1/kernel&variables/1/.ATTRIBUTES/VARIABLE_VALUE
_]
VARIABLE_VALUE#lstm_1/lstm_cell_1/recurrent_kernel&variables/2/.ATTRIBUTES/VARIABLE_VALUE
SQ
VARIABLE_VALUElstm_1/lstm_cell_1/bias&variables/3/.ATTRIBUTES/VARIABLE_VALUE
 
 
 
8
0
1
2
3
4
5
6
7
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 

'0
(1
)2

'0
(1
)2
­
Mlayer_regularization_losses
Nnon_trainable_variables
>regularization_losses
Ometrics
?	variables
@trainable_variables

Players
Qlayer_metrics
 
 
 
 

0
 
 
 
 
 
 
 
 
 
 
 

serving_default_input_2Placeholder*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ*
dtype0*%
shape:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
|
serving_default_input_3Placeholder*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
dtype0*
shape:ÿÿÿÿÿÿÿÿÿ
|
serving_default_input_4Placeholder*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
dtype0*
shape:ÿÿÿÿÿÿÿÿÿ
³
StatefulPartitionedCallStatefulPartitionedCallserving_default_input_2serving_default_input_3serving_default_input_4embedding_1/embeddingslstm_1/lstm_cell_1/kernellstm_1/lstm_cell_1/bias#lstm_1/lstm_cell_1/recurrent_kerneldense/kernel
dense/bias*
Tin
2	*
Tout
2*
_collective_manager_ids
 *\
_output_shapesJ
H:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ7:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ*(
_read_only_resource_inputs

*-
config_proto

CPU

GPU 2J 8 *,
f'R%
#__inference_signature_wrapper_15312
O
saver_filenamePlaceholder*
_output_shapes
: *
dtype0*
shape: 
£
StatefulPartitionedCall_1StatefulPartitionedCallsaver_filename*embedding_1/embeddings/Read/ReadVariableOp dense/kernel/Read/ReadVariableOpdense/bias/Read/ReadVariableOp-lstm_1/lstm_cell_1/kernel/Read/ReadVariableOp7lstm_1/lstm_cell_1/recurrent_kernel/Read/ReadVariableOp+lstm_1/lstm_cell_1/bias/Read/ReadVariableOpConst*
Tin

2*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *'
f"R 
__inference__traced_save_17566
¦
StatefulPartitionedCall_2StatefulPartitionedCallsaver_filenameembedding_1/embeddingsdense/kernel
dense/biaslstm_1/lstm_cell_1/kernel#lstm_1/lstm_cell_1/recurrent_kernellstm_1/lstm_cell_1/bias*
Tin
	2*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 **
f%R#
!__inference__traced_restore_17594ÔÒ$
Ã
Ë
+__inference_lstm_cell_1_layer_call_fn_17521

inputs
states_0
states_1
unknown
	unknown_0
	unknown_1
identity

identity_1

identity_2¢StatefulPartitionedCallÄ
StatefulPartitionedCallStatefulPartitionedCallinputsstates_0states_1unknown	unknown_0	unknown_1*
Tin

2*
Tout
2*
_collective_manager_ids
 *P
_output_shapes>
<:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *O
fJRH
F__inference_lstm_cell_1_layer_call_and_return_conditional_losses_138912
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity

Identity_1Identity StatefulPartitionedCall:output:1^StatefulPartitionedCall*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity_1

Identity_2Identity StatefulPartitionedCall:output:2^StatefulPartitionedCall*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity_2"
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0*[
_input_shapesJ
H:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:::22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs:RN
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
"
_user_specified_name
states/0:RN
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
"
_user_specified_name
states/1
ÿ

B__inference_model_2_layer_call_and_return_conditional_losses_15677
inputs_0
inputs_1
inputs_2&
"embedding_1_embedding_lookup_153184
0lstm_1_lstm_cell_1_split_readvariableop_resource6
2lstm_1_lstm_cell_1_split_1_readvariableop_resource.
*lstm_1_lstm_cell_1_readvariableop_resource+
'dense_tensordot_readvariableop_resource)
%dense_biasadd_readvariableop_resource
identity

identity_1

identity_2¢dense/BiasAdd/ReadVariableOp¢dense/Tensordot/ReadVariableOp¢embedding_1/embedding_lookup¢!lstm_1/lstm_cell_1/ReadVariableOp¢#lstm_1/lstm_cell_1/ReadVariableOp_1¢#lstm_1/lstm_cell_1/ReadVariableOp_2¢#lstm_1/lstm_cell_1/ReadVariableOp_3¢'lstm_1/lstm_cell_1/split/ReadVariableOp¢)lstm_1/lstm_cell_1/split_1/ReadVariableOp¢lstm_1/while
embedding_1/CastCastinputs_0*

DstT0*

SrcT0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2
embedding_1/CastÁ
embedding_1/embedding_lookupResourceGather"embedding_1_embedding_lookup_15318embedding_1/Cast:y:0",/job:localhost/replica:0/task:0/device:CPU:0*
Tindices0*5
_class+
)'loc:@embedding_1/embedding_lookup/15318*5
_output_shapes#
!:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ*
dtype02
embedding_1/embedding_lookup¦
%embedding_1/embedding_lookup/IdentityIdentity%embedding_1/embedding_lookup:output:0",/job:localhost/replica:0/task:0/device:CPU:0*
T0*5
_class+
)'loc:@embedding_1/embedding_lookup/15318*5
_output_shapes#
!:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2'
%embedding_1/embedding_lookup/IdentityÎ
'embedding_1/embedding_lookup/Identity_1Identity.embedding_1/embedding_lookup/Identity:output:0*
T0*5
_output_shapes#
!:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2)
'embedding_1/embedding_lookup/Identity_1u
embedding_1/NotEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *    2
embedding_1/NotEqual/y
embedding_1/NotEqualNotEqualinputs_0embedding_1/NotEqual/y:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2
embedding_1/NotEqual 
activation_1/ReluRelu0embedding_1/embedding_lookup/Identity_1:output:0*
T0*5
_output_shapes#
!:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2
activation_1/Reluw
dropout_1/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *nÛ¶?2
dropout_1/dropout/Const¸
dropout_1/dropout/MulMulactivation_1/Relu:activations:0 dropout_1/dropout/Const:output:0*
T0*5
_output_shapes#
!:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2
dropout_1/dropout/Mul
dropout_1/dropout/ShapeShapeactivation_1/Relu:activations:0*
T0*
_output_shapes
:2
dropout_1/dropout/Shapeà
.dropout_1/dropout/random_uniform/RandomUniformRandomUniform dropout_1/dropout/Shape:output:0*
T0*5
_output_shapes#
!:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ*
dtype020
.dropout_1/dropout/random_uniform/RandomUniform
 dropout_1/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *>2"
 dropout_1/dropout/GreaterEqual/yô
dropout_1/dropout/GreaterEqualGreaterEqual7dropout_1/dropout/random_uniform/RandomUniform:output:0)dropout_1/dropout/GreaterEqual/y:output:0*
T0*5
_output_shapes#
!:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2 
dropout_1/dropout/GreaterEqual«
dropout_1/dropout/CastCast"dropout_1/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*5
_output_shapes#
!:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2
dropout_1/dropout/Cast°
dropout_1/dropout/Mul_1Muldropout_1/dropout/Mul:z:0dropout_1/dropout/Cast:y:0*
T0*5
_output_shapes#
!:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2
dropout_1/dropout/Mul_1
lstm_1/transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
lstm_1/transpose/perm®
lstm_1/transpose	Transposedropout_1/dropout/Mul_1:z:0lstm_1/transpose/perm:output:0*
T0*5
_output_shapes#
!:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2
lstm_1/transpose`
lstm_1/ShapeShapelstm_1/transpose:y:0*
T0*
_output_shapes
:2
lstm_1/Shape
lstm_1/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
lstm_1/strided_slice/stack
lstm_1/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
lstm_1/strided_slice/stack_1
lstm_1/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
lstm_1/strided_slice/stack_2
lstm_1/strided_sliceStridedSlicelstm_1/Shape:output:0#lstm_1/strided_slice/stack:output:0%lstm_1/strided_slice/stack_1:output:0%lstm_1/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
lstm_1/strided_slicey
lstm_1/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ2
lstm_1/ExpandDims/dim­
lstm_1/ExpandDims
ExpandDimsembedding_1/NotEqual:z:0lstm_1/ExpandDims/dim:output:0*
T0
*4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2
lstm_1/ExpandDims
lstm_1/transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
lstm_1/transpose_1/perm²
lstm_1/transpose_1	Transposelstm_1/ExpandDims:output:0 lstm_1/transpose_1/perm:output:0*
T0
*4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2
lstm_1/transpose_1
"lstm_1/TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ2$
"lstm_1/TensorArrayV2/element_shapeÌ
lstm_1/TensorArrayV2TensorListReserve+lstm_1/TensorArrayV2/element_shape:output:0lstm_1/strided_slice:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
lstm_1/TensorArrayV2Í
<lstm_1/TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ   2>
<lstm_1/TensorArrayUnstack/TensorListFromTensor/element_shape
.lstm_1/TensorArrayUnstack/TensorListFromTensorTensorListFromTensorlstm_1/transpose:y:0Elstm_1/TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type020
.lstm_1/TensorArrayUnstack/TensorListFromTensor
lstm_1/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
lstm_1/strided_slice_1/stack
lstm_1/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2 
lstm_1/strided_slice_1/stack_1
lstm_1/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2 
lstm_1/strided_slice_1/stack_2§
lstm_1/strided_slice_1StridedSlicelstm_1/transpose:y:0%lstm_1/strided_slice_1/stack:output:0'lstm_1/strided_slice_1/stack_1:output:0'lstm_1/strided_slice_1/stack_2:output:0*
Index0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
shrink_axis_mask2
lstm_1/strided_slice_1
"lstm_1/lstm_cell_1/ones_like/ShapeShapeinputs_1*
T0*
_output_shapes
:2$
"lstm_1/lstm_cell_1/ones_like/Shape
"lstm_1/lstm_cell_1/ones_like/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ?2$
"lstm_1/lstm_cell_1/ones_like/ConstÑ
lstm_1/lstm_cell_1/ones_likeFill+lstm_1/lstm_cell_1/ones_like/Shape:output:0+lstm_1/lstm_cell_1/ones_like/Const:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
lstm_1/lstm_cell_1/ones_like
 lstm_1/lstm_cell_1/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *ä8?2"
 lstm_1/lstm_cell_1/dropout/ConstÌ
lstm_1/lstm_cell_1/dropout/MulMul%lstm_1/lstm_cell_1/ones_like:output:0)lstm_1/lstm_cell_1/dropout/Const:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2 
lstm_1/lstm_cell_1/dropout/Mul
 lstm_1/lstm_cell_1/dropout/ShapeShape%lstm_1/lstm_cell_1/ones_like:output:0*
T0*
_output_shapes
:2"
 lstm_1/lstm_cell_1/dropout/Shape
7lstm_1/lstm_cell_1/dropout/random_uniform/RandomUniformRandomUniform)lstm_1/lstm_cell_1/dropout/Shape:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
dtype0*
seed±ÿå)*
seed2Î%29
7lstm_1/lstm_cell_1/dropout/random_uniform/RandomUniform
)lstm_1/lstm_cell_1/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *ÍÌÌ=2+
)lstm_1/lstm_cell_1/dropout/GreaterEqual/y
'lstm_1/lstm_cell_1/dropout/GreaterEqualGreaterEqual@lstm_1/lstm_cell_1/dropout/random_uniform/RandomUniform:output:02lstm_1/lstm_cell_1/dropout/GreaterEqual/y:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2)
'lstm_1/lstm_cell_1/dropout/GreaterEqual¹
lstm_1/lstm_cell_1/dropout/CastCast+lstm_1/lstm_cell_1/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2!
lstm_1/lstm_cell_1/dropout/CastÇ
 lstm_1/lstm_cell_1/dropout/Mul_1Mul"lstm_1/lstm_cell_1/dropout/Mul:z:0#lstm_1/lstm_cell_1/dropout/Cast:y:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2"
 lstm_1/lstm_cell_1/dropout/Mul_1
"lstm_1/lstm_cell_1/dropout_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *ä8?2$
"lstm_1/lstm_cell_1/dropout_1/ConstÒ
 lstm_1/lstm_cell_1/dropout_1/MulMul%lstm_1/lstm_cell_1/ones_like:output:0+lstm_1/lstm_cell_1/dropout_1/Const:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2"
 lstm_1/lstm_cell_1/dropout_1/Mul
"lstm_1/lstm_cell_1/dropout_1/ShapeShape%lstm_1/lstm_cell_1/ones_like:output:0*
T0*
_output_shapes
:2$
"lstm_1/lstm_cell_1/dropout_1/Shape
9lstm_1/lstm_cell_1/dropout_1/random_uniform/RandomUniformRandomUniform+lstm_1/lstm_cell_1/dropout_1/Shape:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
dtype0*
seed±ÿå)*
seed2¨Û2;
9lstm_1/lstm_cell_1/dropout_1/random_uniform/RandomUniform
+lstm_1/lstm_cell_1/dropout_1/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *ÍÌÌ=2-
+lstm_1/lstm_cell_1/dropout_1/GreaterEqual/y
)lstm_1/lstm_cell_1/dropout_1/GreaterEqualGreaterEqualBlstm_1/lstm_cell_1/dropout_1/random_uniform/RandomUniform:output:04lstm_1/lstm_cell_1/dropout_1/GreaterEqual/y:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2+
)lstm_1/lstm_cell_1/dropout_1/GreaterEqual¿
!lstm_1/lstm_cell_1/dropout_1/CastCast-lstm_1/lstm_cell_1/dropout_1/GreaterEqual:z:0*

DstT0*

SrcT0
*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2#
!lstm_1/lstm_cell_1/dropout_1/CastÏ
"lstm_1/lstm_cell_1/dropout_1/Mul_1Mul$lstm_1/lstm_cell_1/dropout_1/Mul:z:0%lstm_1/lstm_cell_1/dropout_1/Cast:y:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2$
"lstm_1/lstm_cell_1/dropout_1/Mul_1
"lstm_1/lstm_cell_1/dropout_2/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *ä8?2$
"lstm_1/lstm_cell_1/dropout_2/ConstÒ
 lstm_1/lstm_cell_1/dropout_2/MulMul%lstm_1/lstm_cell_1/ones_like:output:0+lstm_1/lstm_cell_1/dropout_2/Const:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2"
 lstm_1/lstm_cell_1/dropout_2/Mul
"lstm_1/lstm_cell_1/dropout_2/ShapeShape%lstm_1/lstm_cell_1/ones_like:output:0*
T0*
_output_shapes
:2$
"lstm_1/lstm_cell_1/dropout_2/Shape
9lstm_1/lstm_cell_1/dropout_2/random_uniform/RandomUniformRandomUniform+lstm_1/lstm_cell_1/dropout_2/Shape:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
dtype0*
seed±ÿå)*
seed2­Ú2;
9lstm_1/lstm_cell_1/dropout_2/random_uniform/RandomUniform
+lstm_1/lstm_cell_1/dropout_2/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *ÍÌÌ=2-
+lstm_1/lstm_cell_1/dropout_2/GreaterEqual/y
)lstm_1/lstm_cell_1/dropout_2/GreaterEqualGreaterEqualBlstm_1/lstm_cell_1/dropout_2/random_uniform/RandomUniform:output:04lstm_1/lstm_cell_1/dropout_2/GreaterEqual/y:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2+
)lstm_1/lstm_cell_1/dropout_2/GreaterEqual¿
!lstm_1/lstm_cell_1/dropout_2/CastCast-lstm_1/lstm_cell_1/dropout_2/GreaterEqual:z:0*

DstT0*

SrcT0
*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2#
!lstm_1/lstm_cell_1/dropout_2/CastÏ
"lstm_1/lstm_cell_1/dropout_2/Mul_1Mul$lstm_1/lstm_cell_1/dropout_2/Mul:z:0%lstm_1/lstm_cell_1/dropout_2/Cast:y:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2$
"lstm_1/lstm_cell_1/dropout_2/Mul_1
"lstm_1/lstm_cell_1/dropout_3/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *ä8?2$
"lstm_1/lstm_cell_1/dropout_3/ConstÒ
 lstm_1/lstm_cell_1/dropout_3/MulMul%lstm_1/lstm_cell_1/ones_like:output:0+lstm_1/lstm_cell_1/dropout_3/Const:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2"
 lstm_1/lstm_cell_1/dropout_3/Mul
"lstm_1/lstm_cell_1/dropout_3/ShapeShape%lstm_1/lstm_cell_1/ones_like:output:0*
T0*
_output_shapes
:2$
"lstm_1/lstm_cell_1/dropout_3/Shape
9lstm_1/lstm_cell_1/dropout_3/random_uniform/RandomUniformRandomUniform+lstm_1/lstm_cell_1/dropout_3/Shape:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
dtype0*
seed±ÿå)*
seed2×2;
9lstm_1/lstm_cell_1/dropout_3/random_uniform/RandomUniform
+lstm_1/lstm_cell_1/dropout_3/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *ÍÌÌ=2-
+lstm_1/lstm_cell_1/dropout_3/GreaterEqual/y
)lstm_1/lstm_cell_1/dropout_3/GreaterEqualGreaterEqualBlstm_1/lstm_cell_1/dropout_3/random_uniform/RandomUniform:output:04lstm_1/lstm_cell_1/dropout_3/GreaterEqual/y:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2+
)lstm_1/lstm_cell_1/dropout_3/GreaterEqual¿
!lstm_1/lstm_cell_1/dropout_3/CastCast-lstm_1/lstm_cell_1/dropout_3/GreaterEqual:z:0*

DstT0*

SrcT0
*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2#
!lstm_1/lstm_cell_1/dropout_3/CastÏ
"lstm_1/lstm_cell_1/dropout_3/Mul_1Mul$lstm_1/lstm_cell_1/dropout_3/Mul:z:0%lstm_1/lstm_cell_1/dropout_3/Cast:y:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2$
"lstm_1/lstm_cell_1/dropout_3/Mul_1v
lstm_1/lstm_cell_1/ConstConst*
_output_shapes
: *
dtype0*
value	B :2
lstm_1/lstm_cell_1/Const
"lstm_1/lstm_cell_1/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2$
"lstm_1/lstm_cell_1/split/split_dimÅ
'lstm_1/lstm_cell_1/split/ReadVariableOpReadVariableOp0lstm_1_lstm_cell_1_split_readvariableop_resource* 
_output_shapes
:
*
dtype02)
'lstm_1/lstm_cell_1/split/ReadVariableOpû
lstm_1/lstm_cell_1/splitSplit+lstm_1/lstm_cell_1/split/split_dim:output:0/lstm_1/lstm_cell_1/split/ReadVariableOp:value:0*
T0*D
_output_shapes2
0:
:
:
:
*
	num_split2
lstm_1/lstm_cell_1/split·
lstm_1/lstm_cell_1/MatMulMatMullstm_1/strided_slice_1:output:0!lstm_1/lstm_cell_1/split:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
lstm_1/lstm_cell_1/MatMul»
lstm_1/lstm_cell_1/MatMul_1MatMullstm_1/strided_slice_1:output:0!lstm_1/lstm_cell_1/split:output:1*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
lstm_1/lstm_cell_1/MatMul_1»
lstm_1/lstm_cell_1/MatMul_2MatMullstm_1/strided_slice_1:output:0!lstm_1/lstm_cell_1/split:output:2*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
lstm_1/lstm_cell_1/MatMul_2»
lstm_1/lstm_cell_1/MatMul_3MatMullstm_1/strided_slice_1:output:0!lstm_1/lstm_cell_1/split:output:3*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
lstm_1/lstm_cell_1/MatMul_3z
lstm_1/lstm_cell_1/Const_1Const*
_output_shapes
: *
dtype0*
value	B :2
lstm_1/lstm_cell_1/Const_1
$lstm_1/lstm_cell_1/split_1/split_dimConst*
_output_shapes
: *
dtype0*
value	B : 2&
$lstm_1/lstm_cell_1/split_1/split_dimÆ
)lstm_1/lstm_cell_1/split_1/ReadVariableOpReadVariableOp2lstm_1_lstm_cell_1_split_1_readvariableop_resource*
_output_shapes	
:*
dtype02+
)lstm_1/lstm_cell_1/split_1/ReadVariableOpï
lstm_1/lstm_cell_1/split_1Split-lstm_1/lstm_cell_1/split_1/split_dim:output:01lstm_1/lstm_cell_1/split_1/ReadVariableOp:value:0*
T0*0
_output_shapes
::::*
	num_split2
lstm_1/lstm_cell_1/split_1À
lstm_1/lstm_cell_1/BiasAddBiasAdd#lstm_1/lstm_cell_1/MatMul:product:0#lstm_1/lstm_cell_1/split_1:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
lstm_1/lstm_cell_1/BiasAddÆ
lstm_1/lstm_cell_1/BiasAdd_1BiasAdd%lstm_1/lstm_cell_1/MatMul_1:product:0#lstm_1/lstm_cell_1/split_1:output:1*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
lstm_1/lstm_cell_1/BiasAdd_1Æ
lstm_1/lstm_cell_1/BiasAdd_2BiasAdd%lstm_1/lstm_cell_1/MatMul_2:product:0#lstm_1/lstm_cell_1/split_1:output:2*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
lstm_1/lstm_cell_1/BiasAdd_2Æ
lstm_1/lstm_cell_1/BiasAdd_3BiasAdd%lstm_1/lstm_cell_1/MatMul_3:product:0#lstm_1/lstm_cell_1/split_1:output:3*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
lstm_1/lstm_cell_1/BiasAdd_3
lstm_1/lstm_cell_1/mulMulinputs_1$lstm_1/lstm_cell_1/dropout/Mul_1:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
lstm_1/lstm_cell_1/mul 
lstm_1/lstm_cell_1/mul_1Mulinputs_1&lstm_1/lstm_cell_1/dropout_1/Mul_1:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
lstm_1/lstm_cell_1/mul_1 
lstm_1/lstm_cell_1/mul_2Mulinputs_1&lstm_1/lstm_cell_1/dropout_2/Mul_1:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
lstm_1/lstm_cell_1/mul_2 
lstm_1/lstm_cell_1/mul_3Mulinputs_1&lstm_1/lstm_cell_1/dropout_3/Mul_1:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
lstm_1/lstm_cell_1/mul_3³
!lstm_1/lstm_cell_1/ReadVariableOpReadVariableOp*lstm_1_lstm_cell_1_readvariableop_resource* 
_output_shapes
:
*
dtype02#
!lstm_1/lstm_cell_1/ReadVariableOp¡
&lstm_1/lstm_cell_1/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB"        2(
&lstm_1/lstm_cell_1/strided_slice/stack¥
(lstm_1/lstm_cell_1/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       2*
(lstm_1/lstm_cell_1/strided_slice/stack_1¥
(lstm_1/lstm_cell_1/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2*
(lstm_1/lstm_cell_1/strided_slice/stack_2ð
 lstm_1/lstm_cell_1/strided_sliceStridedSlice)lstm_1/lstm_cell_1/ReadVariableOp:value:0/lstm_1/lstm_cell_1/strided_slice/stack:output:01lstm_1/lstm_cell_1/strided_slice/stack_1:output:01lstm_1/lstm_cell_1/strided_slice/stack_2:output:0*
Index0*
T0* 
_output_shapes
:
*

begin_mask*
end_mask2"
 lstm_1/lstm_cell_1/strided_slice¾
lstm_1/lstm_cell_1/MatMul_4MatMullstm_1/lstm_cell_1/mul:z:0)lstm_1/lstm_cell_1/strided_slice:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
lstm_1/lstm_cell_1/MatMul_4¸
lstm_1/lstm_cell_1/addAddV2#lstm_1/lstm_cell_1/BiasAdd:output:0%lstm_1/lstm_cell_1/MatMul_4:product:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
lstm_1/lstm_cell_1/add
lstm_1/lstm_cell_1/SigmoidSigmoidlstm_1/lstm_cell_1/add:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
lstm_1/lstm_cell_1/Sigmoid·
#lstm_1/lstm_cell_1/ReadVariableOp_1ReadVariableOp*lstm_1_lstm_cell_1_readvariableop_resource* 
_output_shapes
:
*
dtype02%
#lstm_1/lstm_cell_1/ReadVariableOp_1¥
(lstm_1/lstm_cell_1/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB"       2*
(lstm_1/lstm_cell_1/strided_slice_1/stack©
*lstm_1/lstm_cell_1/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       2,
*lstm_1/lstm_cell_1/strided_slice_1/stack_1©
*lstm_1/lstm_cell_1/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2,
*lstm_1/lstm_cell_1/strided_slice_1/stack_2ü
"lstm_1/lstm_cell_1/strided_slice_1StridedSlice+lstm_1/lstm_cell_1/ReadVariableOp_1:value:01lstm_1/lstm_cell_1/strided_slice_1/stack:output:03lstm_1/lstm_cell_1/strided_slice_1/stack_1:output:03lstm_1/lstm_cell_1/strided_slice_1/stack_2:output:0*
Index0*
T0* 
_output_shapes
:
*

begin_mask*
end_mask2$
"lstm_1/lstm_cell_1/strided_slice_1Â
lstm_1/lstm_cell_1/MatMul_5MatMullstm_1/lstm_cell_1/mul_1:z:0+lstm_1/lstm_cell_1/strided_slice_1:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
lstm_1/lstm_cell_1/MatMul_5¾
lstm_1/lstm_cell_1/add_1AddV2%lstm_1/lstm_cell_1/BiasAdd_1:output:0%lstm_1/lstm_cell_1/MatMul_5:product:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
lstm_1/lstm_cell_1/add_1
lstm_1/lstm_cell_1/Sigmoid_1Sigmoidlstm_1/lstm_cell_1/add_1:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
lstm_1/lstm_cell_1/Sigmoid_1
lstm_1/lstm_cell_1/mul_4Mul lstm_1/lstm_cell_1/Sigmoid_1:y:0inputs_2*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
lstm_1/lstm_cell_1/mul_4·
#lstm_1/lstm_cell_1/ReadVariableOp_2ReadVariableOp*lstm_1_lstm_cell_1_readvariableop_resource* 
_output_shapes
:
*
dtype02%
#lstm_1/lstm_cell_1/ReadVariableOp_2¥
(lstm_1/lstm_cell_1/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB"       2*
(lstm_1/lstm_cell_1/strided_slice_2/stack©
*lstm_1/lstm_cell_1/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       2,
*lstm_1/lstm_cell_1/strided_slice_2/stack_1©
*lstm_1/lstm_cell_1/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2,
*lstm_1/lstm_cell_1/strided_slice_2/stack_2ü
"lstm_1/lstm_cell_1/strided_slice_2StridedSlice+lstm_1/lstm_cell_1/ReadVariableOp_2:value:01lstm_1/lstm_cell_1/strided_slice_2/stack:output:03lstm_1/lstm_cell_1/strided_slice_2/stack_1:output:03lstm_1/lstm_cell_1/strided_slice_2/stack_2:output:0*
Index0*
T0* 
_output_shapes
:
*

begin_mask*
end_mask2$
"lstm_1/lstm_cell_1/strided_slice_2Â
lstm_1/lstm_cell_1/MatMul_6MatMullstm_1/lstm_cell_1/mul_2:z:0+lstm_1/lstm_cell_1/strided_slice_2:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
lstm_1/lstm_cell_1/MatMul_6¾
lstm_1/lstm_cell_1/add_2AddV2%lstm_1/lstm_cell_1/BiasAdd_2:output:0%lstm_1/lstm_cell_1/MatMul_6:product:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
lstm_1/lstm_cell_1/add_2
lstm_1/lstm_cell_1/TanhTanhlstm_1/lstm_cell_1/add_2:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
lstm_1/lstm_cell_1/Tanh«
lstm_1/lstm_cell_1/mul_5Mullstm_1/lstm_cell_1/Sigmoid:y:0lstm_1/lstm_cell_1/Tanh:y:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
lstm_1/lstm_cell_1/mul_5¬
lstm_1/lstm_cell_1/add_3AddV2lstm_1/lstm_cell_1/mul_4:z:0lstm_1/lstm_cell_1/mul_5:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
lstm_1/lstm_cell_1/add_3·
#lstm_1/lstm_cell_1/ReadVariableOp_3ReadVariableOp*lstm_1_lstm_cell_1_readvariableop_resource* 
_output_shapes
:
*
dtype02%
#lstm_1/lstm_cell_1/ReadVariableOp_3¥
(lstm_1/lstm_cell_1/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB"       2*
(lstm_1/lstm_cell_1/strided_slice_3/stack©
*lstm_1/lstm_cell_1/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        2,
*lstm_1/lstm_cell_1/strided_slice_3/stack_1©
*lstm_1/lstm_cell_1/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2,
*lstm_1/lstm_cell_1/strided_slice_3/stack_2ü
"lstm_1/lstm_cell_1/strided_slice_3StridedSlice+lstm_1/lstm_cell_1/ReadVariableOp_3:value:01lstm_1/lstm_cell_1/strided_slice_3/stack:output:03lstm_1/lstm_cell_1/strided_slice_3/stack_1:output:03lstm_1/lstm_cell_1/strided_slice_3/stack_2:output:0*
Index0*
T0* 
_output_shapes
:
*

begin_mask*
end_mask2$
"lstm_1/lstm_cell_1/strided_slice_3Â
lstm_1/lstm_cell_1/MatMul_7MatMullstm_1/lstm_cell_1/mul_3:z:0+lstm_1/lstm_cell_1/strided_slice_3:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
lstm_1/lstm_cell_1/MatMul_7¾
lstm_1/lstm_cell_1/add_4AddV2%lstm_1/lstm_cell_1/BiasAdd_3:output:0%lstm_1/lstm_cell_1/MatMul_7:product:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
lstm_1/lstm_cell_1/add_4
lstm_1/lstm_cell_1/Sigmoid_2Sigmoidlstm_1/lstm_cell_1/add_4:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
lstm_1/lstm_cell_1/Sigmoid_2
lstm_1/lstm_cell_1/Tanh_1Tanhlstm_1/lstm_cell_1/add_3:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
lstm_1/lstm_cell_1/Tanh_1¯
lstm_1/lstm_cell_1/mul_6Mul lstm_1/lstm_cell_1/Sigmoid_2:y:0lstm_1/lstm_cell_1/Tanh_1:y:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
lstm_1/lstm_cell_1/mul_6
$lstm_1/TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ   2&
$lstm_1/TensorArrayV2_1/element_shapeÒ
lstm_1/TensorArrayV2_1TensorListReserve-lstm_1/TensorArrayV2_1/element_shape:output:0lstm_1/strided_slice:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
lstm_1/TensorArrayV2_1\
lstm_1/timeConst*
_output_shapes
: *
dtype0*
value	B : 2
lstm_1/time
$lstm_1/TensorArrayV2_2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ2&
$lstm_1/TensorArrayV2_2/element_shapeÒ
lstm_1/TensorArrayV2_2TensorListReserve-lstm_1/TensorArrayV2_2/element_shape:output:0lstm_1/strided_slice:output:0*
_output_shapes
: *
element_dtype0
*

shape_type02
lstm_1/TensorArrayV2_2Ñ
>lstm_1/TensorArrayUnstack_1/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ   2@
>lstm_1/TensorArrayUnstack_1/TensorListFromTensor/element_shape
0lstm_1/TensorArrayUnstack_1/TensorListFromTensorTensorListFromTensorlstm_1/transpose_1:y:0Glstm_1/TensorArrayUnstack_1/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0
*

shape_type022
0lstm_1/TensorArrayUnstack_1/TensorListFromTensor
lstm_1/zeros_like	ZerosLikelstm_1/lstm_cell_1/mul_6:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
lstm_1/zeros_like
lstm_1/while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ2!
lstm_1/while/maximum_iterationsx
lstm_1/while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : 2
lstm_1/while/loop_counterµ
lstm_1/whileWhile"lstm_1/while/loop_counter:output:0(lstm_1/while/maximum_iterations:output:0lstm_1/time:output:0lstm_1/TensorArrayV2_1:handle:0lstm_1/zeros_like:y:0inputs_1inputs_2lstm_1/strided_slice:output:0>lstm_1/TensorArrayUnstack/TensorListFromTensor:output_handle:0@lstm_1/TensorArrayUnstack_1/TensorListFromTensor:output_handle:00lstm_1_lstm_cell_1_split_readvariableop_resource2lstm_1_lstm_cell_1_split_1_readvariableop_resource*lstm_1_lstm_cell_1_readvariableop_resource*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*d
_output_shapesR
P: : : : :ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ: : : : : : *%
_read_only_resource_inputs

*#
bodyR
lstm_1_while_body_15463*#
condR
lstm_1_while_cond_15462*c
output_shapesR
P: : : : :ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ: : : : : : *
parallel_iterations 2
lstm_1/whileÃ
7lstm_1/TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ   29
7lstm_1/TensorArrayV2Stack/TensorListStack/element_shape
)lstm_1/TensorArrayV2Stack/TensorListStackTensorListStacklstm_1/while:output:3@lstm_1/TensorArrayV2Stack/TensorListStack/element_shape:output:0*5
_output_shapes#
!:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ*
element_dtype02+
)lstm_1/TensorArrayV2Stack/TensorListStack
lstm_1/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB:
ÿÿÿÿÿÿÿÿÿ2
lstm_1/strided_slice_2/stack
lstm_1/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2 
lstm_1/strided_slice_2/stack_1
lstm_1/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2 
lstm_1/strided_slice_2/stack_2Å
lstm_1/strided_slice_2StridedSlice2lstm_1/TensorArrayV2Stack/TensorListStack:tensor:0%lstm_1/strided_slice_2/stack:output:0'lstm_1/strided_slice_2/stack_1:output:0'lstm_1/strided_slice_2/stack_2:output:0*
Index0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
shrink_axis_mask2
lstm_1/strided_slice_2
lstm_1/transpose_2/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
lstm_1/transpose_2/permË
lstm_1/transpose_2	Transpose2lstm_1/TensorArrayV2Stack/TensorListStack:tensor:0 lstm_1/transpose_2/perm:output:0*
T0*5
_output_shapes#
!:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2
lstm_1/transpose_2t
lstm_1/runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    2
lstm_1/runtime©
dense/Tensordot/ReadVariableOpReadVariableOp'dense_tensordot_readvariableop_resource*
_output_shapes
:	7*
dtype02 
dense/Tensordot/ReadVariableOpv
dense/Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:2
dense/Tensordot/axes}
dense/Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       2
dense/Tensordot/freet
dense/Tensordot/ShapeShapelstm_1/transpose_2:y:0*
T0*
_output_shapes
:2
dense/Tensordot/Shape
dense/Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : 2
dense/Tensordot/GatherV2/axisï
dense/Tensordot/GatherV2GatherV2dense/Tensordot/Shape:output:0dense/Tensordot/free:output:0&dense/Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2
dense/Tensordot/GatherV2
dense/Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2!
dense/Tensordot/GatherV2_1/axisõ
dense/Tensordot/GatherV2_1GatherV2dense/Tensordot/Shape:output:0dense/Tensordot/axes:output:0(dense/Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2
dense/Tensordot/GatherV2_1x
dense/Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2
dense/Tensordot/Const
dense/Tensordot/ProdProd!dense/Tensordot/GatherV2:output:0dense/Tensordot/Const:output:0*
T0*
_output_shapes
: 2
dense/Tensordot/Prod|
dense/Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: 2
dense/Tensordot/Const_1 
dense/Tensordot/Prod_1Prod#dense/Tensordot/GatherV2_1:output:0 dense/Tensordot/Const_1:output:0*
T0*
_output_shapes
: 2
dense/Tensordot/Prod_1|
dense/Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 2
dense/Tensordot/concat/axisÎ
dense/Tensordot/concatConcatV2dense/Tensordot/free:output:0dense/Tensordot/axes:output:0$dense/Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:2
dense/Tensordot/concat¤
dense/Tensordot/stackPackdense/Tensordot/Prod:output:0dense/Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:2
dense/Tensordot/stack¼
dense/Tensordot/transpose	Transposelstm_1/transpose_2:y:0dense/Tensordot/concat:output:0*
T0*5
_output_shapes#
!:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2
dense/Tensordot/transpose·
dense/Tensordot/ReshapeReshapedense/Tensordot/transpose:y:0dense/Tensordot/stack:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2
dense/Tensordot/Reshape¶
dense/Tensordot/MatMulMatMul dense/Tensordot/Reshape:output:0&dense/Tensordot/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ72
dense/Tensordot/MatMul|
dense/Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:72
dense/Tensordot/Const_2
dense/Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2
dense/Tensordot/concat_1/axisÛ
dense/Tensordot/concat_1ConcatV2!dense/Tensordot/GatherV2:output:0 dense/Tensordot/Const_2:output:0&dense/Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:2
dense/Tensordot/concat_1±
dense/TensordotReshape dense/Tensordot/MatMul:product:0!dense/Tensordot/concat_1:output:0*
T0*4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ72
dense/Tensordot
dense/BiasAdd/ReadVariableOpReadVariableOp%dense_biasadd_readvariableop_resource*
_output_shapes
:7*
dtype02
dense/BiasAdd/ReadVariableOp¨
dense/BiasAddBiasAdddense/Tensordot:output:0$dense/BiasAdd/ReadVariableOp:value:0*
T0*4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ72
dense/BiasAdd
dense/Max/reduction_indicesConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ2
dense/Max/reduction_indices«
	dense/MaxMaxdense/BiasAdd:output:0$dense/Max/reduction_indices:output:0*
T0*4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ*
	keep_dims(2
	dense/Max
	dense/subSubdense/BiasAdd:output:0dense/Max:output:0*
T0*4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ72
	dense/subk
	dense/ExpExpdense/sub:z:0*
T0*4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ72
	dense/Exp
dense/Sum/reduction_indicesConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ2
dense/Sum/reduction_indices¢
	dense/SumSumdense/Exp:y:0$dense/Sum/reduction_indices:output:0*
T0*4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ*
	keep_dims(2
	dense/Sum
dense/truedivRealDivdense/Exp:y:0dense/Sum:output:0*
T0*4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ72
dense/truedivÌ
IdentityIdentitydense/truediv:z:0^dense/BiasAdd/ReadVariableOp^dense/Tensordot/ReadVariableOp^embedding_1/embedding_lookup"^lstm_1/lstm_cell_1/ReadVariableOp$^lstm_1/lstm_cell_1/ReadVariableOp_1$^lstm_1/lstm_cell_1/ReadVariableOp_2$^lstm_1/lstm_cell_1/ReadVariableOp_3(^lstm_1/lstm_cell_1/split/ReadVariableOp*^lstm_1/lstm_cell_1/split_1/ReadVariableOp^lstm_1/while*
T0*4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ72

IdentityÈ

Identity_1Identitylstm_1/while:output:5^dense/BiasAdd/ReadVariableOp^dense/Tensordot/ReadVariableOp^embedding_1/embedding_lookup"^lstm_1/lstm_cell_1/ReadVariableOp$^lstm_1/lstm_cell_1/ReadVariableOp_1$^lstm_1/lstm_cell_1/ReadVariableOp_2$^lstm_1/lstm_cell_1/ReadVariableOp_3(^lstm_1/lstm_cell_1/split/ReadVariableOp*^lstm_1/lstm_cell_1/split_1/ReadVariableOp^lstm_1/while*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity_1È

Identity_2Identitylstm_1/while:output:6^dense/BiasAdd/ReadVariableOp^dense/Tensordot/ReadVariableOp^embedding_1/embedding_lookup"^lstm_1/lstm_cell_1/ReadVariableOp$^lstm_1/lstm_cell_1/ReadVariableOp_1$^lstm_1/lstm_cell_1/ReadVariableOp_2$^lstm_1/lstm_cell_1/ReadVariableOp_3(^lstm_1/lstm_cell_1/split/ReadVariableOp*^lstm_1/lstm_cell_1/split_1/ReadVariableOp^lstm_1/while*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity_2"
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0*o
_input_shapes^
\:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ::::::2<
dense/BiasAdd/ReadVariableOpdense/BiasAdd/ReadVariableOp2@
dense/Tensordot/ReadVariableOpdense/Tensordot/ReadVariableOp2<
embedding_1/embedding_lookupembedding_1/embedding_lookup2F
!lstm_1/lstm_cell_1/ReadVariableOp!lstm_1/lstm_cell_1/ReadVariableOp2J
#lstm_1/lstm_cell_1/ReadVariableOp_1#lstm_1/lstm_cell_1/ReadVariableOp_12J
#lstm_1/lstm_cell_1/ReadVariableOp_2#lstm_1/lstm_cell_1/ReadVariableOp_22J
#lstm_1/lstm_cell_1/ReadVariableOp_3#lstm_1/lstm_cell_1/ReadVariableOp_32R
'lstm_1/lstm_cell_1/split/ReadVariableOp'lstm_1/lstm_cell_1/split/ReadVariableOp2V
)lstm_1/lstm_cell_1/split_1/ReadVariableOp)lstm_1/lstm_cell_1/split_1/ReadVariableOp2
lstm_1/whilelstm_1/while:Z V
0
_output_shapes
:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
"
_user_specified_name
inputs/0:RN
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
"
_user_specified_name
inputs/1:RN
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
"
_user_specified_name
inputs/2

°
__inference__traced_save_17566
file_prefix5
1savev2_embedding_1_embeddings_read_readvariableop+
'savev2_dense_kernel_read_readvariableop)
%savev2_dense_bias_read_readvariableop8
4savev2_lstm_1_lstm_cell_1_kernel_read_readvariableopB
>savev2_lstm_1_lstm_cell_1_recurrent_kernel_read_readvariableop6
2savev2_lstm_1_lstm_cell_1_bias_read_readvariableop
savev2_const

identity_1¢MergeV2Checkpoints
StaticRegexFullMatchStaticRegexFullMatchfile_prefix"/device:CPU:**
_output_shapes
: *
pattern
^s3://.*2
StaticRegexFullMatchc
ConstConst"/device:CPU:**
_output_shapes
: *
dtype0*
valueB B.part2
Constl
Const_1Const"/device:CPU:**
_output_shapes
: *
dtype0*
valueB B
_temp/part2	
Const_1
SelectSelectStaticRegexFullMatch:output:0Const:output:0Const_1:output:0"/device:CPU:**
T0*
_output_shapes
: 2
Selectt

StringJoin
StringJoinfile_prefixSelect:output:0"/device:CPU:**
N*
_output_shapes
: 2

StringJoinZ

num_shardsConst*
_output_shapes
: *
dtype0*
value	B :2

num_shards
ShardedFilename/shardConst"/device:CPU:0*
_output_shapes
: *
dtype0*
value	B : 2
ShardedFilename/shard¦
ShardedFilenameShardedFilenameStringJoin:output:0ShardedFilename/shard:output:0num_shards:output:0"/device:CPU:0*
_output_shapes
: 2
ShardedFilenameÃ
SaveV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:*
dtype0*Õ
valueËBÈB:layer_with_weights-0/embeddings/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUEB&variables/1/.ATTRIBUTES/VARIABLE_VALUEB&variables/2/.ATTRIBUTES/VARIABLE_VALUEB&variables/3/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH2
SaveV2/tensor_names
SaveV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:*
dtype0*!
valueBB B B B B B B 2
SaveV2/shape_and_slicesí
SaveV2SaveV2ShardedFilename:filename:0SaveV2/tensor_names:output:0 SaveV2/shape_and_slices:output:01savev2_embedding_1_embeddings_read_readvariableop'savev2_dense_kernel_read_readvariableop%savev2_dense_bias_read_readvariableop4savev2_lstm_1_lstm_cell_1_kernel_read_readvariableop>savev2_lstm_1_lstm_cell_1_recurrent_kernel_read_readvariableop2savev2_lstm_1_lstm_cell_1_bias_read_readvariableopsavev2_const"/device:CPU:0*
_output_shapes
 *
dtypes
	22
SaveV2º
&MergeV2Checkpoints/checkpoint_prefixesPackShardedFilename:filename:0^SaveV2"/device:CPU:0*
N*
T0*
_output_shapes
:2(
&MergeV2Checkpoints/checkpoint_prefixes¡
MergeV2CheckpointsMergeV2Checkpoints/MergeV2Checkpoints/checkpoint_prefixes:output:0file_prefix"/device:CPU:0*
_output_shapes
 2
MergeV2Checkpointsr
IdentityIdentityfile_prefix^MergeV2Checkpoints"/device:CPU:0*
T0*
_output_shapes
: 2

Identitym

Identity_1IdentityIdentity:output:0^MergeV2Checkpoints*
T0*
_output_shapes
: 2

Identity_1"!

identity_1Identity_1:output:0*R
_input_shapesA
?: :	7:	7:7:
:
:: 2(
MergeV2CheckpointsMergeV2Checkpoints:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix:%!

_output_shapes
:	7:%!

_output_shapes
:	7: 

_output_shapes
:7:&"
 
_output_shapes
:
:&"
 
_output_shapes
:
:!

_output_shapes	
::

_output_shapes
: 
ó
c
D__inference_dropout_1_layer_call_and_return_conditional_losses_14466

inputs
identityc
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *nÛ¶?2
dropout/Const
dropout/MulMulinputsdropout/Const:output:0*
T0*5
_output_shapes#
!:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2
dropout/MulT
dropout/ShapeShapeinputs*
T0*
_output_shapes
:2
dropout/ShapeÂ
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*5
_output_shapes#
!:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ*
dtype02&
$dropout/random_uniform/RandomUniformu
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *>2
dropout/GreaterEqual/yÌ
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*5
_output_shapes#
!:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2
dropout/GreaterEqual
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*5
_output_shapes#
!:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2
dropout/Cast
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*5
_output_shapes#
!:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2
dropout/Mul_1s
IdentityIdentitydropout/Mul_1:z:0*
T0*5
_output_shapes#
!:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*4
_input_shapes#
!:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ:] Y
5
_output_shapes#
!:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
Ü	

while_cond_17069
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_placeholder_4
while_less_strided_slice3
/while_while_cond_17069___redundant_placeholder03
/while_while_cond_17069___redundant_placeholder13
/while_while_cond_17069___redundant_placeholder23
/while_while_cond_17069___redundant_placeholder33
/while_while_cond_17069___redundant_placeholder4
while_identity
n

while/LessLesswhile_placeholderwhile_less_strided_slice*
T0*
_output_shapes
: 2

while/Less]
while/IdentityIdentitywhile/Less:z:0*
T0
*
_output_shapes
: 2
while/Identity")
while_identitywhile/Identity:output:0*m
_input_shapes\
Z: : : : :ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ: :::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :.*
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ:.*
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ:.*
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ:

_output_shapes
: :

_output_shapes
::	

_output_shapes
:
Ã
Ë
+__inference_lstm_cell_1_layer_call_fn_17504

inputs
states_0
states_1
unknown
	unknown_0
	unknown_1
identity

identity_1

identity_2¢StatefulPartitionedCallÄ
StatefulPartitionedCallStatefulPartitionedCallinputsstates_0states_1unknown	unknown_0	unknown_1*
Tin

2*
Tout
2*
_collective_manager_ids
 *P
_output_shapes>
<:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *O
fJRH
F__inference_lstm_cell_1_layer_call_and_return_conditional_losses_138142
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity

Identity_1Identity StatefulPartitionedCall:output:1^StatefulPartitionedCall*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity_1

Identity_2Identity StatefulPartitionedCall:output:2^StatefulPartitionedCall*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity_2"
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0*[
_input_shapesJ
H:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:::22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs:RN
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
"
_user_specified_name
states/0:RN
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
"
_user_specified_name
states/1
ÿ
b
D__inference_dropout_1_layer_call_and_return_conditional_losses_16061

inputs

identity_1h
IdentityIdentityinputs*
T0*5
_output_shapes#
!:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2

Identityw

Identity_1IdentityIdentity:output:0*
T0*5
_output_shapes#
!:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2

Identity_1"!

identity_1Identity_1:output:0*4
_input_shapes#
!:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ:] Y
5
_output_shapes#
!:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
õ"

B__inference_model_2_layer_call_and_return_conditional_losses_15216

inputs
inputs_1
inputs_2
embedding_1_15192
lstm_1_15199
lstm_1_15201
lstm_1_15203
dense_15208
dense_15210
identity

identity_1

identity_2¢dense/StatefulPartitionedCall¢!dropout_1/StatefulPartitionedCall¢#embedding_1/StatefulPartitionedCall¢lstm_1/StatefulPartitionedCall
#embedding_1/StatefulPartitionedCallStatefulPartitionedCallinputsembedding_1_15192*
Tin
2*
Tout
2*
_collective_manager_ids
 *5
_output_shapes#
!:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ*#
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *O
fJRH
F__inference_embedding_1_layer_call_and_return_conditional_losses_144272%
#embedding_1/StatefulPartitionedCallu
embedding_1/NotEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *    2
embedding_1/NotEqual/y
embedding_1/NotEqualNotEqualinputsembedding_1/NotEqual/y:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2
embedding_1/NotEqual
activation_1/PartitionedCallPartitionedCall,embedding_1/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *5
_output_shapes#
!:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *P
fKRI
G__inference_activation_1_layer_call_and_return_conditional_losses_144462
activation_1/PartitionedCall
!dropout_1/StatefulPartitionedCallStatefulPartitionedCall%activation_1/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *5
_output_shapes#
!:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *M
fHRF
D__inference_dropout_1_layer_call_and_return_conditional_losses_144662#
!dropout_1/StatefulPartitionedCall¤
lstm_1/StatefulPartitionedCallStatefulPartitionedCall*dropout_1/StatefulPartitionedCall:output:0embedding_1/NotEqual:z:0inputs_1inputs_2lstm_1_15199lstm_1_15201lstm_1_15203*
Tin
	2
*
Tout
2*
_collective_manager_ids
 *]
_output_shapesK
I:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *J
fERC
A__inference_lstm_1_layer_call_and_return_conditional_losses_147992 
lstm_1/StatefulPartitionedCall°
dense/StatefulPartitionedCallStatefulPartitionedCall'lstm_1/StatefulPartitionedCall:output:0dense_15208dense_15210*
Tin
2*
Tout
2*
_collective_manager_ids
 *4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ7*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *I
fDRB
@__inference_dense_layer_call_and_return_conditional_losses_151342
dense/StatefulPartitionedCall
IdentityIdentity&dense/StatefulPartitionedCall:output:0^dense/StatefulPartitionedCall"^dropout_1/StatefulPartitionedCall$^embedding_1/StatefulPartitionedCall^lstm_1/StatefulPartitionedCall*
T0*4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ72

Identity

Identity_1Identity'lstm_1/StatefulPartitionedCall:output:1^dense/StatefulPartitionedCall"^dropout_1/StatefulPartitionedCall$^embedding_1/StatefulPartitionedCall^lstm_1/StatefulPartitionedCall*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity_1

Identity_2Identity'lstm_1/StatefulPartitionedCall:output:2^dense/StatefulPartitionedCall"^dropout_1/StatefulPartitionedCall$^embedding_1/StatefulPartitionedCall^lstm_1/StatefulPartitionedCall*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity_2"
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0*o
_input_shapes^
\:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ::::::2>
dense/StatefulPartitionedCalldense/StatefulPartitionedCall2F
!dropout_1/StatefulPartitionedCall!dropout_1/StatefulPartitionedCall2J
#embedding_1/StatefulPartitionedCall#embedding_1/StatefulPartitionedCall2@
lstm_1/StatefulPartitionedCalllstm_1/StatefulPartitionedCall:X T
0
_output_shapes
:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs:PL
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs:PL
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
Ý&
ß
@__inference_dense_layer_call_and_return_conditional_losses_15134

inputs%
!tensordot_readvariableop_resource#
biasadd_readvariableop_resource
identity¢BiasAdd/ReadVariableOp¢Tensordot/ReadVariableOp
Tensordot/ReadVariableOpReadVariableOp!tensordot_readvariableop_resource*
_output_shapes
:	7*
dtype02
Tensordot/ReadVariableOpj
Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:2
Tensordot/axesq
Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       2
Tensordot/freeX
Tensordot/ShapeShapeinputs*
T0*
_output_shapes
:2
Tensordot/Shapet
Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : 2
Tensordot/GatherV2/axisÑ
Tensordot/GatherV2GatherV2Tensordot/Shape:output:0Tensordot/free:output:0 Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2
Tensordot/GatherV2x
Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2
Tensordot/GatherV2_1/axis×
Tensordot/GatherV2_1GatherV2Tensordot/Shape:output:0Tensordot/axes:output:0"Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2
Tensordot/GatherV2_1l
Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2
Tensordot/Const
Tensordot/ProdProdTensordot/GatherV2:output:0Tensordot/Const:output:0*
T0*
_output_shapes
: 2
Tensordot/Prodp
Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: 2
Tensordot/Const_1
Tensordot/Prod_1ProdTensordot/GatherV2_1:output:0Tensordot/Const_1:output:0*
T0*
_output_shapes
: 2
Tensordot/Prod_1p
Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 2
Tensordot/concat/axis°
Tensordot/concatConcatV2Tensordot/free:output:0Tensordot/axes:output:0Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:2
Tensordot/concat
Tensordot/stackPackTensordot/Prod:output:0Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:2
Tensordot/stack
Tensordot/transpose	TransposeinputsTensordot/concat:output:0*
T0*5
_output_shapes#
!:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2
Tensordot/transpose
Tensordot/ReshapeReshapeTensordot/transpose:y:0Tensordot/stack:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2
Tensordot/Reshape
Tensordot/MatMulMatMulTensordot/Reshape:output:0 Tensordot/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ72
Tensordot/MatMulp
Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:72
Tensordot/Const_2t
Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2
Tensordot/concat_1/axis½
Tensordot/concat_1ConcatV2Tensordot/GatherV2:output:0Tensordot/Const_2:output:0 Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:2
Tensordot/concat_1
	TensordotReshapeTensordot/MatMul:product:0Tensordot/concat_1:output:0*
T0*4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ72
	Tensordot
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:7*
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddTensordot:output:0BiasAdd/ReadVariableOp:value:0*
T0*4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ72	
BiasAddy
Max/reduction_indicesConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ2
Max/reduction_indices
MaxMaxBiasAdd:output:0Max/reduction_indices:output:0*
T0*4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ*
	keep_dims(2
Maxp
subSubBiasAdd:output:0Max:output:0*
T0*4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ72
subY
ExpExpsub:z:0*
T0*4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ72
Expy
Sum/reduction_indicesConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ2
Sum/reduction_indices
SumSumExp:y:0Sum/reduction_indices:output:0*
T0*4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ*
	keep_dims(2
Sums
truedivRealDivExp:y:0Sum:output:0*
T0*4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ72	
truediv 
IdentityIdentitytruediv:z:0^BiasAdd/ReadVariableOp^Tensordot/ReadVariableOp*
T0*4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ72

Identity"
identityIdentity:output:0*<
_input_shapes+
):ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp24
Tensordot/ReadVariableOpTensordot/ReadVariableOp:] Y
5
_output_shapes#
!:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
Ð
¨

while_body_14618
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_placeholder_4
while_strided_slice_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0[
Wwhile_tensorarrayv2read_1_tensorlistgetitem_tensorarrayunstack_1_tensorlistfromtensor_05
1while_lstm_cell_1_split_readvariableop_resource_07
3while_lstm_cell_1_split_1_readvariableop_resource_0/
+while_lstm_cell_1_readvariableop_resource_0
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_identity_5
while_identity_6
while_strided_sliceU
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorY
Uwhile_tensorarrayv2read_1_tensorlistgetitem_tensorarrayunstack_1_tensorlistfromtensor3
/while_lstm_cell_1_split_readvariableop_resource5
1while_lstm_cell_1_split_1_readvariableop_resource-
)while_lstm_cell_1_readvariableop_resource¢ while/lstm_cell_1/ReadVariableOp¢"while/lstm_cell_1/ReadVariableOp_1¢"while/lstm_cell_1/ReadVariableOp_2¢"while/lstm_cell_1/ReadVariableOp_3¢&while/lstm_cell_1/split/ReadVariableOp¢(while/lstm_cell_1/split_1/ReadVariableOpÃ
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ   29
7while/TensorArrayV2Read/TensorListGetItem/element_shapeÔ
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
element_dtype02+
)while/TensorArrayV2Read/TensorListGetItemÇ
9while/TensorArrayV2Read_1/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ   2;
9while/TensorArrayV2Read_1/TensorListGetItem/element_shapeÝ
+while/TensorArrayV2Read_1/TensorListGetItemTensorListGetItemWwhile_tensorarrayv2read_1_tensorlistgetitem_tensorarrayunstack_1_tensorlistfromtensor_0while_placeholderBwhile/TensorArrayV2Read_1/TensorListGetItem/element_shape:output:0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
element_dtype0
2-
+while/TensorArrayV2Read_1/TensorListGetItem
!while/lstm_cell_1/ones_like/ShapeShapewhile_placeholder_3*
T0*
_output_shapes
:2#
!while/lstm_cell_1/ones_like/Shape
!while/lstm_cell_1/ones_like/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ?2#
!while/lstm_cell_1/ones_like/ConstÍ
while/lstm_cell_1/ones_likeFill*while/lstm_cell_1/ones_like/Shape:output:0*while/lstm_cell_1/ones_like/Const:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
while/lstm_cell_1/ones_like
while/lstm_cell_1/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *ä8?2!
while/lstm_cell_1/dropout/ConstÈ
while/lstm_cell_1/dropout/MulMul$while/lstm_cell_1/ones_like:output:0(while/lstm_cell_1/dropout/Const:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
while/lstm_cell_1/dropout/Mul
while/lstm_cell_1/dropout/ShapeShape$while/lstm_cell_1/ones_like:output:0*
T0*
_output_shapes
:2!
while/lstm_cell_1/dropout/Shape
6while/lstm_cell_1/dropout/random_uniform/RandomUniformRandomUniform(while/lstm_cell_1/dropout/Shape:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
dtype0*
seed±ÿå)*
seed2¢íÛ28
6while/lstm_cell_1/dropout/random_uniform/RandomUniform
(while/lstm_cell_1/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *ÍÌÌ=2*
(while/lstm_cell_1/dropout/GreaterEqual/y
&while/lstm_cell_1/dropout/GreaterEqualGreaterEqual?while/lstm_cell_1/dropout/random_uniform/RandomUniform:output:01while/lstm_cell_1/dropout/GreaterEqual/y:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2(
&while/lstm_cell_1/dropout/GreaterEqual¶
while/lstm_cell_1/dropout/CastCast*while/lstm_cell_1/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2 
while/lstm_cell_1/dropout/CastÃ
while/lstm_cell_1/dropout/Mul_1Mul!while/lstm_cell_1/dropout/Mul:z:0"while/lstm_cell_1/dropout/Cast:y:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2!
while/lstm_cell_1/dropout/Mul_1
!while/lstm_cell_1/dropout_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *ä8?2#
!while/lstm_cell_1/dropout_1/ConstÎ
while/lstm_cell_1/dropout_1/MulMul$while/lstm_cell_1/ones_like:output:0*while/lstm_cell_1/dropout_1/Const:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2!
while/lstm_cell_1/dropout_1/Mul
!while/lstm_cell_1/dropout_1/ShapeShape$while/lstm_cell_1/ones_like:output:0*
T0*
_output_shapes
:2#
!while/lstm_cell_1/dropout_1/Shape
8while/lstm_cell_1/dropout_1/random_uniform/RandomUniformRandomUniform*while/lstm_cell_1/dropout_1/Shape:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
dtype0*
seed±ÿå)*
seed2»ÂÀ2:
8while/lstm_cell_1/dropout_1/random_uniform/RandomUniform
*while/lstm_cell_1/dropout_1/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *ÍÌÌ=2,
*while/lstm_cell_1/dropout_1/GreaterEqual/y
(while/lstm_cell_1/dropout_1/GreaterEqualGreaterEqualAwhile/lstm_cell_1/dropout_1/random_uniform/RandomUniform:output:03while/lstm_cell_1/dropout_1/GreaterEqual/y:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2*
(while/lstm_cell_1/dropout_1/GreaterEqual¼
 while/lstm_cell_1/dropout_1/CastCast,while/lstm_cell_1/dropout_1/GreaterEqual:z:0*

DstT0*

SrcT0
*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2"
 while/lstm_cell_1/dropout_1/CastË
!while/lstm_cell_1/dropout_1/Mul_1Mul#while/lstm_cell_1/dropout_1/Mul:z:0$while/lstm_cell_1/dropout_1/Cast:y:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2#
!while/lstm_cell_1/dropout_1/Mul_1
!while/lstm_cell_1/dropout_2/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *ä8?2#
!while/lstm_cell_1/dropout_2/ConstÎ
while/lstm_cell_1/dropout_2/MulMul$while/lstm_cell_1/ones_like:output:0*while/lstm_cell_1/dropout_2/Const:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2!
while/lstm_cell_1/dropout_2/Mul
!while/lstm_cell_1/dropout_2/ShapeShape$while/lstm_cell_1/ones_like:output:0*
T0*
_output_shapes
:2#
!while/lstm_cell_1/dropout_2/Shape
8while/lstm_cell_1/dropout_2/random_uniform/RandomUniformRandomUniform*while/lstm_cell_1/dropout_2/Shape:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
dtype0*
seed±ÿå)*
seed2Õ¹2:
8while/lstm_cell_1/dropout_2/random_uniform/RandomUniform
*while/lstm_cell_1/dropout_2/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *ÍÌÌ=2,
*while/lstm_cell_1/dropout_2/GreaterEqual/y
(while/lstm_cell_1/dropout_2/GreaterEqualGreaterEqualAwhile/lstm_cell_1/dropout_2/random_uniform/RandomUniform:output:03while/lstm_cell_1/dropout_2/GreaterEqual/y:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2*
(while/lstm_cell_1/dropout_2/GreaterEqual¼
 while/lstm_cell_1/dropout_2/CastCast,while/lstm_cell_1/dropout_2/GreaterEqual:z:0*

DstT0*

SrcT0
*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2"
 while/lstm_cell_1/dropout_2/CastË
!while/lstm_cell_1/dropout_2/Mul_1Mul#while/lstm_cell_1/dropout_2/Mul:z:0$while/lstm_cell_1/dropout_2/Cast:y:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2#
!while/lstm_cell_1/dropout_2/Mul_1
!while/lstm_cell_1/dropout_3/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *ä8?2#
!while/lstm_cell_1/dropout_3/ConstÎ
while/lstm_cell_1/dropout_3/MulMul$while/lstm_cell_1/ones_like:output:0*while/lstm_cell_1/dropout_3/Const:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2!
while/lstm_cell_1/dropout_3/Mul
!while/lstm_cell_1/dropout_3/ShapeShape$while/lstm_cell_1/ones_like:output:0*
T0*
_output_shapes
:2#
!while/lstm_cell_1/dropout_3/Shape
8while/lstm_cell_1/dropout_3/random_uniform/RandomUniformRandomUniform*while/lstm_cell_1/dropout_3/Shape:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
dtype0*
seed±ÿå)*
seed2½Üü2:
8while/lstm_cell_1/dropout_3/random_uniform/RandomUniform
*while/lstm_cell_1/dropout_3/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *ÍÌÌ=2,
*while/lstm_cell_1/dropout_3/GreaterEqual/y
(while/lstm_cell_1/dropout_3/GreaterEqualGreaterEqualAwhile/lstm_cell_1/dropout_3/random_uniform/RandomUniform:output:03while/lstm_cell_1/dropout_3/GreaterEqual/y:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2*
(while/lstm_cell_1/dropout_3/GreaterEqual¼
 while/lstm_cell_1/dropout_3/CastCast,while/lstm_cell_1/dropout_3/GreaterEqual:z:0*

DstT0*

SrcT0
*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2"
 while/lstm_cell_1/dropout_3/CastË
!while/lstm_cell_1/dropout_3/Mul_1Mul#while/lstm_cell_1/dropout_3/Mul:z:0$while/lstm_cell_1/dropout_3/Cast:y:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2#
!while/lstm_cell_1/dropout_3/Mul_1t
while/lstm_cell_1/ConstConst*
_output_shapes
: *
dtype0*
value	B :2
while/lstm_cell_1/Const
!while/lstm_cell_1/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2#
!while/lstm_cell_1/split/split_dimÄ
&while/lstm_cell_1/split/ReadVariableOpReadVariableOp1while_lstm_cell_1_split_readvariableop_resource_0* 
_output_shapes
:
*
dtype02(
&while/lstm_cell_1/split/ReadVariableOp÷
while/lstm_cell_1/splitSplit*while/lstm_cell_1/split/split_dim:output:0.while/lstm_cell_1/split/ReadVariableOp:value:0*
T0*D
_output_shapes2
0:
:
:
:
*
	num_split2
while/lstm_cell_1/splitÅ
while/lstm_cell_1/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:0 while/lstm_cell_1/split:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
while/lstm_cell_1/MatMulÉ
while/lstm_cell_1/MatMul_1MatMul0while/TensorArrayV2Read/TensorListGetItem:item:0 while/lstm_cell_1/split:output:1*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
while/lstm_cell_1/MatMul_1É
while/lstm_cell_1/MatMul_2MatMul0while/TensorArrayV2Read/TensorListGetItem:item:0 while/lstm_cell_1/split:output:2*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
while/lstm_cell_1/MatMul_2É
while/lstm_cell_1/MatMul_3MatMul0while/TensorArrayV2Read/TensorListGetItem:item:0 while/lstm_cell_1/split:output:3*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
while/lstm_cell_1/MatMul_3x
while/lstm_cell_1/Const_1Const*
_output_shapes
: *
dtype0*
value	B :2
while/lstm_cell_1/Const_1
#while/lstm_cell_1/split_1/split_dimConst*
_output_shapes
: *
dtype0*
value	B : 2%
#while/lstm_cell_1/split_1/split_dimÅ
(while/lstm_cell_1/split_1/ReadVariableOpReadVariableOp3while_lstm_cell_1_split_1_readvariableop_resource_0*
_output_shapes	
:*
dtype02*
(while/lstm_cell_1/split_1/ReadVariableOpë
while/lstm_cell_1/split_1Split,while/lstm_cell_1/split_1/split_dim:output:00while/lstm_cell_1/split_1/ReadVariableOp:value:0*
T0*0
_output_shapes
::::*
	num_split2
while/lstm_cell_1/split_1¼
while/lstm_cell_1/BiasAddBiasAdd"while/lstm_cell_1/MatMul:product:0"while/lstm_cell_1/split_1:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
while/lstm_cell_1/BiasAddÂ
while/lstm_cell_1/BiasAdd_1BiasAdd$while/lstm_cell_1/MatMul_1:product:0"while/lstm_cell_1/split_1:output:1*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
while/lstm_cell_1/BiasAdd_1Â
while/lstm_cell_1/BiasAdd_2BiasAdd$while/lstm_cell_1/MatMul_2:product:0"while/lstm_cell_1/split_1:output:2*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
while/lstm_cell_1/BiasAdd_2Â
while/lstm_cell_1/BiasAdd_3BiasAdd$while/lstm_cell_1/MatMul_3:product:0"while/lstm_cell_1/split_1:output:3*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
while/lstm_cell_1/BiasAdd_3¢
while/lstm_cell_1/mulMulwhile_placeholder_3#while/lstm_cell_1/dropout/Mul_1:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
while/lstm_cell_1/mul¨
while/lstm_cell_1/mul_1Mulwhile_placeholder_3%while/lstm_cell_1/dropout_1/Mul_1:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
while/lstm_cell_1/mul_1¨
while/lstm_cell_1/mul_2Mulwhile_placeholder_3%while/lstm_cell_1/dropout_2/Mul_1:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
while/lstm_cell_1/mul_2¨
while/lstm_cell_1/mul_3Mulwhile_placeholder_3%while/lstm_cell_1/dropout_3/Mul_1:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
while/lstm_cell_1/mul_3²
 while/lstm_cell_1/ReadVariableOpReadVariableOp+while_lstm_cell_1_readvariableop_resource_0* 
_output_shapes
:
*
dtype02"
 while/lstm_cell_1/ReadVariableOp
%while/lstm_cell_1/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB"        2'
%while/lstm_cell_1/strided_slice/stack£
'while/lstm_cell_1/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       2)
'while/lstm_cell_1/strided_slice/stack_1£
'while/lstm_cell_1/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2)
'while/lstm_cell_1/strided_slice/stack_2ê
while/lstm_cell_1/strided_sliceStridedSlice(while/lstm_cell_1/ReadVariableOp:value:0.while/lstm_cell_1/strided_slice/stack:output:00while/lstm_cell_1/strided_slice/stack_1:output:00while/lstm_cell_1/strided_slice/stack_2:output:0*
Index0*
T0* 
_output_shapes
:
*

begin_mask*
end_mask2!
while/lstm_cell_1/strided_sliceº
while/lstm_cell_1/MatMul_4MatMulwhile/lstm_cell_1/mul:z:0(while/lstm_cell_1/strided_slice:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
while/lstm_cell_1/MatMul_4´
while/lstm_cell_1/addAddV2"while/lstm_cell_1/BiasAdd:output:0$while/lstm_cell_1/MatMul_4:product:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
while/lstm_cell_1/add
while/lstm_cell_1/SigmoidSigmoidwhile/lstm_cell_1/add:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
while/lstm_cell_1/Sigmoid¶
"while/lstm_cell_1/ReadVariableOp_1ReadVariableOp+while_lstm_cell_1_readvariableop_resource_0* 
_output_shapes
:
*
dtype02$
"while/lstm_cell_1/ReadVariableOp_1£
'while/lstm_cell_1/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB"       2)
'while/lstm_cell_1/strided_slice_1/stack§
)while/lstm_cell_1/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       2+
)while/lstm_cell_1/strided_slice_1/stack_1§
)while/lstm_cell_1/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2+
)while/lstm_cell_1/strided_slice_1/stack_2ö
!while/lstm_cell_1/strided_slice_1StridedSlice*while/lstm_cell_1/ReadVariableOp_1:value:00while/lstm_cell_1/strided_slice_1/stack:output:02while/lstm_cell_1/strided_slice_1/stack_1:output:02while/lstm_cell_1/strided_slice_1/stack_2:output:0*
Index0*
T0* 
_output_shapes
:
*

begin_mask*
end_mask2#
!while/lstm_cell_1/strided_slice_1¾
while/lstm_cell_1/MatMul_5MatMulwhile/lstm_cell_1/mul_1:z:0*while/lstm_cell_1/strided_slice_1:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
while/lstm_cell_1/MatMul_5º
while/lstm_cell_1/add_1AddV2$while/lstm_cell_1/BiasAdd_1:output:0$while/lstm_cell_1/MatMul_5:product:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
while/lstm_cell_1/add_1
while/lstm_cell_1/Sigmoid_1Sigmoidwhile/lstm_cell_1/add_1:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
while/lstm_cell_1/Sigmoid_1¢
while/lstm_cell_1/mul_4Mulwhile/lstm_cell_1/Sigmoid_1:y:0while_placeholder_4*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
while/lstm_cell_1/mul_4¶
"while/lstm_cell_1/ReadVariableOp_2ReadVariableOp+while_lstm_cell_1_readvariableop_resource_0* 
_output_shapes
:
*
dtype02$
"while/lstm_cell_1/ReadVariableOp_2£
'while/lstm_cell_1/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB"       2)
'while/lstm_cell_1/strided_slice_2/stack§
)while/lstm_cell_1/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       2+
)while/lstm_cell_1/strided_slice_2/stack_1§
)while/lstm_cell_1/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2+
)while/lstm_cell_1/strided_slice_2/stack_2ö
!while/lstm_cell_1/strided_slice_2StridedSlice*while/lstm_cell_1/ReadVariableOp_2:value:00while/lstm_cell_1/strided_slice_2/stack:output:02while/lstm_cell_1/strided_slice_2/stack_1:output:02while/lstm_cell_1/strided_slice_2/stack_2:output:0*
Index0*
T0* 
_output_shapes
:
*

begin_mask*
end_mask2#
!while/lstm_cell_1/strided_slice_2¾
while/lstm_cell_1/MatMul_6MatMulwhile/lstm_cell_1/mul_2:z:0*while/lstm_cell_1/strided_slice_2:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
while/lstm_cell_1/MatMul_6º
while/lstm_cell_1/add_2AddV2$while/lstm_cell_1/BiasAdd_2:output:0$while/lstm_cell_1/MatMul_6:product:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
while/lstm_cell_1/add_2
while/lstm_cell_1/TanhTanhwhile/lstm_cell_1/add_2:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
while/lstm_cell_1/Tanh§
while/lstm_cell_1/mul_5Mulwhile/lstm_cell_1/Sigmoid:y:0while/lstm_cell_1/Tanh:y:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
while/lstm_cell_1/mul_5¨
while/lstm_cell_1/add_3AddV2while/lstm_cell_1/mul_4:z:0while/lstm_cell_1/mul_5:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
while/lstm_cell_1/add_3¶
"while/lstm_cell_1/ReadVariableOp_3ReadVariableOp+while_lstm_cell_1_readvariableop_resource_0* 
_output_shapes
:
*
dtype02$
"while/lstm_cell_1/ReadVariableOp_3£
'while/lstm_cell_1/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB"       2)
'while/lstm_cell_1/strided_slice_3/stack§
)while/lstm_cell_1/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        2+
)while/lstm_cell_1/strided_slice_3/stack_1§
)while/lstm_cell_1/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2+
)while/lstm_cell_1/strided_slice_3/stack_2ö
!while/lstm_cell_1/strided_slice_3StridedSlice*while/lstm_cell_1/ReadVariableOp_3:value:00while/lstm_cell_1/strided_slice_3/stack:output:02while/lstm_cell_1/strided_slice_3/stack_1:output:02while/lstm_cell_1/strided_slice_3/stack_2:output:0*
Index0*
T0* 
_output_shapes
:
*

begin_mask*
end_mask2#
!while/lstm_cell_1/strided_slice_3¾
while/lstm_cell_1/MatMul_7MatMulwhile/lstm_cell_1/mul_3:z:0*while/lstm_cell_1/strided_slice_3:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
while/lstm_cell_1/MatMul_7º
while/lstm_cell_1/add_4AddV2$while/lstm_cell_1/BiasAdd_3:output:0$while/lstm_cell_1/MatMul_7:product:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
while/lstm_cell_1/add_4
while/lstm_cell_1/Sigmoid_2Sigmoidwhile/lstm_cell_1/add_4:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
while/lstm_cell_1/Sigmoid_2
while/lstm_cell_1/Tanh_1Tanhwhile/lstm_cell_1/add_3:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
while/lstm_cell_1/Tanh_1«
while/lstm_cell_1/mul_6Mulwhile/lstm_cell_1/Sigmoid_2:y:0while/lstm_cell_1/Tanh_1:y:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
while/lstm_cell_1/mul_6}
while/Tile/multiplesConst*
_output_shapes
:*
dtype0*
valueB"      2
while/Tile/multiples¥

while/TileTile2while/TensorArrayV2Read_1/TensorListGetItem:item:0while/Tile/multiples:output:0*
T0
*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

while/Tile¦
while/SelectV2SelectV2while/Tile:output:0while/lstm_cell_1/mul_6:z:0while_placeholder_2*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
while/SelectV2
while/Tile_1/multiplesConst*
_output_shapes
:*
dtype0*
valueB"      2
while/Tile_1/multiples«
while/Tile_1Tile2while/TensorArrayV2Read_1/TensorListGetItem:item:0while/Tile_1/multiples:output:0*
T0
*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
while/Tile_1
while/Tile_2/multiplesConst*
_output_shapes
:*
dtype0*
valueB"      2
while/Tile_2/multiples«
while/Tile_2Tile2while/TensorArrayV2Read_1/TensorListGetItem:item:0while/Tile_2/multiples:output:0*
T0
*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
while/Tile_2¬
while/SelectV2_1SelectV2while/Tile_1:output:0while/lstm_cell_1/mul_6:z:0while_placeholder_3*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
while/SelectV2_1¬
while/SelectV2_2SelectV2while/Tile_2:output:0while/lstm_cell_1/add_3:z:0while_placeholder_4*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
while/SelectV2_2Û
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholderwhile/SelectV2:output:0*
_output_shapes
: *
element_dtype02,
*while/TensorArrayV2Write/TensorListSetItem\
while/add/yConst*
_output_shapes
: *
dtype0*
value	B :2
while/add/yi
	while/addAddV2while_placeholderwhile/add/y:output:0*
T0*
_output_shapes
: 2
	while/add`
while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :2
while/add_1/yv
while/add_1AddV2while_while_loop_counterwhile/add_1/y:output:0*
T0*
_output_shapes
: 2
while/add_1Ä
while/IdentityIdentitywhile/add_1:z:0!^while/lstm_cell_1/ReadVariableOp#^while/lstm_cell_1/ReadVariableOp_1#^while/lstm_cell_1/ReadVariableOp_2#^while/lstm_cell_1/ReadVariableOp_3'^while/lstm_cell_1/split/ReadVariableOp)^while/lstm_cell_1/split_1/ReadVariableOp*
T0*
_output_shapes
: 2
while/Identity×
while/Identity_1Identitywhile_while_maximum_iterations!^while/lstm_cell_1/ReadVariableOp#^while/lstm_cell_1/ReadVariableOp_1#^while/lstm_cell_1/ReadVariableOp_2#^while/lstm_cell_1/ReadVariableOp_3'^while/lstm_cell_1/split/ReadVariableOp)^while/lstm_cell_1/split_1/ReadVariableOp*
T0*
_output_shapes
: 2
while/Identity_1Æ
while/Identity_2Identitywhile/add:z:0!^while/lstm_cell_1/ReadVariableOp#^while/lstm_cell_1/ReadVariableOp_1#^while/lstm_cell_1/ReadVariableOp_2#^while/lstm_cell_1/ReadVariableOp_3'^while/lstm_cell_1/split/ReadVariableOp)^while/lstm_cell_1/split_1/ReadVariableOp*
T0*
_output_shapes
: 2
while/Identity_2ó
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0!^while/lstm_cell_1/ReadVariableOp#^while/lstm_cell_1/ReadVariableOp_1#^while/lstm_cell_1/ReadVariableOp_2#^while/lstm_cell_1/ReadVariableOp_3'^while/lstm_cell_1/split/ReadVariableOp)^while/lstm_cell_1/split_1/ReadVariableOp*
T0*
_output_shapes
: 2
while/Identity_3â
while/Identity_4Identitywhile/SelectV2:output:0!^while/lstm_cell_1/ReadVariableOp#^while/lstm_cell_1/ReadVariableOp_1#^while/lstm_cell_1/ReadVariableOp_2#^while/lstm_cell_1/ReadVariableOp_3'^while/lstm_cell_1/split/ReadVariableOp)^while/lstm_cell_1/split_1/ReadVariableOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
while/Identity_4ä
while/Identity_5Identitywhile/SelectV2_1:output:0!^while/lstm_cell_1/ReadVariableOp#^while/lstm_cell_1/ReadVariableOp_1#^while/lstm_cell_1/ReadVariableOp_2#^while/lstm_cell_1/ReadVariableOp_3'^while/lstm_cell_1/split/ReadVariableOp)^while/lstm_cell_1/split_1/ReadVariableOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
while/Identity_5ä
while/Identity_6Identitywhile/SelectV2_2:output:0!^while/lstm_cell_1/ReadVariableOp#^while/lstm_cell_1/ReadVariableOp_1#^while/lstm_cell_1/ReadVariableOp_2#^while/lstm_cell_1/ReadVariableOp_3'^while/lstm_cell_1/split/ReadVariableOp)^while/lstm_cell_1/split_1/ReadVariableOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
while/Identity_6")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"-
while_identity_5while/Identity_5:output:0"-
while_identity_6while/Identity_6:output:0"X
)while_lstm_cell_1_readvariableop_resource+while_lstm_cell_1_readvariableop_resource_0"h
1while_lstm_cell_1_split_1_readvariableop_resource3while_lstm_cell_1_split_1_readvariableop_resource_0"d
/while_lstm_cell_1_split_readvariableop_resource1while_lstm_cell_1_split_readvariableop_resource_0",
while_strided_slicewhile_strided_slice_0"°
Uwhile_tensorarrayv2read_1_tensorlistgetitem_tensorarrayunstack_1_tensorlistfromtensorWwhile_tensorarrayv2read_1_tensorlistgetitem_tensorarrayunstack_1_tensorlistfromtensor_0"¨
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*i
_input_shapesX
V: : : : :ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ: : : :::2D
 while/lstm_cell_1/ReadVariableOp while/lstm_cell_1/ReadVariableOp2H
"while/lstm_cell_1/ReadVariableOp_1"while/lstm_cell_1/ReadVariableOp_12H
"while/lstm_cell_1/ReadVariableOp_2"while/lstm_cell_1/ReadVariableOp_22H
"while/lstm_cell_1/ReadVariableOp_3"while/lstm_cell_1/ReadVariableOp_32P
&while/lstm_cell_1/split/ReadVariableOp&while/lstm_cell_1/split/ReadVariableOp2T
(while/lstm_cell_1/split_1/ReadVariableOp(while/lstm_cell_1/split_1/ReadVariableOp: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :.*
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ:.*
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ:.*
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ:

_output_shapes
: :

_output_shapes
: :	

_output_shapes
: 
Ð
¨

while_body_16786
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_placeholder_4
while_strided_slice_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0[
Wwhile_tensorarrayv2read_1_tensorlistgetitem_tensorarrayunstack_1_tensorlistfromtensor_05
1while_lstm_cell_1_split_readvariableop_resource_07
3while_lstm_cell_1_split_1_readvariableop_resource_0/
+while_lstm_cell_1_readvariableop_resource_0
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_identity_5
while_identity_6
while_strided_sliceU
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorY
Uwhile_tensorarrayv2read_1_tensorlistgetitem_tensorarrayunstack_1_tensorlistfromtensor3
/while_lstm_cell_1_split_readvariableop_resource5
1while_lstm_cell_1_split_1_readvariableop_resource-
)while_lstm_cell_1_readvariableop_resource¢ while/lstm_cell_1/ReadVariableOp¢"while/lstm_cell_1/ReadVariableOp_1¢"while/lstm_cell_1/ReadVariableOp_2¢"while/lstm_cell_1/ReadVariableOp_3¢&while/lstm_cell_1/split/ReadVariableOp¢(while/lstm_cell_1/split_1/ReadVariableOpÃ
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ   29
7while/TensorArrayV2Read/TensorListGetItem/element_shapeÔ
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
element_dtype02+
)while/TensorArrayV2Read/TensorListGetItemÇ
9while/TensorArrayV2Read_1/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ   2;
9while/TensorArrayV2Read_1/TensorListGetItem/element_shapeÝ
+while/TensorArrayV2Read_1/TensorListGetItemTensorListGetItemWwhile_tensorarrayv2read_1_tensorlistgetitem_tensorarrayunstack_1_tensorlistfromtensor_0while_placeholderBwhile/TensorArrayV2Read_1/TensorListGetItem/element_shape:output:0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
element_dtype0
2-
+while/TensorArrayV2Read_1/TensorListGetItem
!while/lstm_cell_1/ones_like/ShapeShapewhile_placeholder_3*
T0*
_output_shapes
:2#
!while/lstm_cell_1/ones_like/Shape
!while/lstm_cell_1/ones_like/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ?2#
!while/lstm_cell_1/ones_like/ConstÍ
while/lstm_cell_1/ones_likeFill*while/lstm_cell_1/ones_like/Shape:output:0*while/lstm_cell_1/ones_like/Const:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
while/lstm_cell_1/ones_like
while/lstm_cell_1/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *ä8?2!
while/lstm_cell_1/dropout/ConstÈ
while/lstm_cell_1/dropout/MulMul$while/lstm_cell_1/ones_like:output:0(while/lstm_cell_1/dropout/Const:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
while/lstm_cell_1/dropout/Mul
while/lstm_cell_1/dropout/ShapeShape$while/lstm_cell_1/ones_like:output:0*
T0*
_output_shapes
:2!
while/lstm_cell_1/dropout/Shape
6while/lstm_cell_1/dropout/random_uniform/RandomUniformRandomUniform(while/lstm_cell_1/dropout/Shape:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
dtype0*
seed±ÿå)*
seed2ß·28
6while/lstm_cell_1/dropout/random_uniform/RandomUniform
(while/lstm_cell_1/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *ÍÌÌ=2*
(while/lstm_cell_1/dropout/GreaterEqual/y
&while/lstm_cell_1/dropout/GreaterEqualGreaterEqual?while/lstm_cell_1/dropout/random_uniform/RandomUniform:output:01while/lstm_cell_1/dropout/GreaterEqual/y:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2(
&while/lstm_cell_1/dropout/GreaterEqual¶
while/lstm_cell_1/dropout/CastCast*while/lstm_cell_1/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2 
while/lstm_cell_1/dropout/CastÃ
while/lstm_cell_1/dropout/Mul_1Mul!while/lstm_cell_1/dropout/Mul:z:0"while/lstm_cell_1/dropout/Cast:y:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2!
while/lstm_cell_1/dropout/Mul_1
!while/lstm_cell_1/dropout_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *ä8?2#
!while/lstm_cell_1/dropout_1/ConstÎ
while/lstm_cell_1/dropout_1/MulMul$while/lstm_cell_1/ones_like:output:0*while/lstm_cell_1/dropout_1/Const:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2!
while/lstm_cell_1/dropout_1/Mul
!while/lstm_cell_1/dropout_1/ShapeShape$while/lstm_cell_1/ones_like:output:0*
T0*
_output_shapes
:2#
!while/lstm_cell_1/dropout_1/Shape
8while/lstm_cell_1/dropout_1/random_uniform/RandomUniformRandomUniform*while/lstm_cell_1/dropout_1/Shape:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
dtype0*
seed±ÿå)*
seed2ýÕ¹2:
8while/lstm_cell_1/dropout_1/random_uniform/RandomUniform
*while/lstm_cell_1/dropout_1/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *ÍÌÌ=2,
*while/lstm_cell_1/dropout_1/GreaterEqual/y
(while/lstm_cell_1/dropout_1/GreaterEqualGreaterEqualAwhile/lstm_cell_1/dropout_1/random_uniform/RandomUniform:output:03while/lstm_cell_1/dropout_1/GreaterEqual/y:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2*
(while/lstm_cell_1/dropout_1/GreaterEqual¼
 while/lstm_cell_1/dropout_1/CastCast,while/lstm_cell_1/dropout_1/GreaterEqual:z:0*

DstT0*

SrcT0
*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2"
 while/lstm_cell_1/dropout_1/CastË
!while/lstm_cell_1/dropout_1/Mul_1Mul#while/lstm_cell_1/dropout_1/Mul:z:0$while/lstm_cell_1/dropout_1/Cast:y:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2#
!while/lstm_cell_1/dropout_1/Mul_1
!while/lstm_cell_1/dropout_2/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *ä8?2#
!while/lstm_cell_1/dropout_2/ConstÎ
while/lstm_cell_1/dropout_2/MulMul$while/lstm_cell_1/ones_like:output:0*while/lstm_cell_1/dropout_2/Const:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2!
while/lstm_cell_1/dropout_2/Mul
!while/lstm_cell_1/dropout_2/ShapeShape$while/lstm_cell_1/ones_like:output:0*
T0*
_output_shapes
:2#
!while/lstm_cell_1/dropout_2/Shape
8while/lstm_cell_1/dropout_2/random_uniform/RandomUniformRandomUniform*while/lstm_cell_1/dropout_2/Shape:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
dtype0*
seed±ÿå)*
seed2¶2:
8while/lstm_cell_1/dropout_2/random_uniform/RandomUniform
*while/lstm_cell_1/dropout_2/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *ÍÌÌ=2,
*while/lstm_cell_1/dropout_2/GreaterEqual/y
(while/lstm_cell_1/dropout_2/GreaterEqualGreaterEqualAwhile/lstm_cell_1/dropout_2/random_uniform/RandomUniform:output:03while/lstm_cell_1/dropout_2/GreaterEqual/y:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2*
(while/lstm_cell_1/dropout_2/GreaterEqual¼
 while/lstm_cell_1/dropout_2/CastCast,while/lstm_cell_1/dropout_2/GreaterEqual:z:0*

DstT0*

SrcT0
*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2"
 while/lstm_cell_1/dropout_2/CastË
!while/lstm_cell_1/dropout_2/Mul_1Mul#while/lstm_cell_1/dropout_2/Mul:z:0$while/lstm_cell_1/dropout_2/Cast:y:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2#
!while/lstm_cell_1/dropout_2/Mul_1
!while/lstm_cell_1/dropout_3/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *ä8?2#
!while/lstm_cell_1/dropout_3/ConstÎ
while/lstm_cell_1/dropout_3/MulMul$while/lstm_cell_1/ones_like:output:0*while/lstm_cell_1/dropout_3/Const:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2!
while/lstm_cell_1/dropout_3/Mul
!while/lstm_cell_1/dropout_3/ShapeShape$while/lstm_cell_1/ones_like:output:0*
T0*
_output_shapes
:2#
!while/lstm_cell_1/dropout_3/Shape
8while/lstm_cell_1/dropout_3/random_uniform/RandomUniformRandomUniform*while/lstm_cell_1/dropout_3/Shape:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
dtype0*
seed±ÿå)*
seed2ím2:
8while/lstm_cell_1/dropout_3/random_uniform/RandomUniform
*while/lstm_cell_1/dropout_3/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *ÍÌÌ=2,
*while/lstm_cell_1/dropout_3/GreaterEqual/y
(while/lstm_cell_1/dropout_3/GreaterEqualGreaterEqualAwhile/lstm_cell_1/dropout_3/random_uniform/RandomUniform:output:03while/lstm_cell_1/dropout_3/GreaterEqual/y:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2*
(while/lstm_cell_1/dropout_3/GreaterEqual¼
 while/lstm_cell_1/dropout_3/CastCast,while/lstm_cell_1/dropout_3/GreaterEqual:z:0*

DstT0*

SrcT0
*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2"
 while/lstm_cell_1/dropout_3/CastË
!while/lstm_cell_1/dropout_3/Mul_1Mul#while/lstm_cell_1/dropout_3/Mul:z:0$while/lstm_cell_1/dropout_3/Cast:y:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2#
!while/lstm_cell_1/dropout_3/Mul_1t
while/lstm_cell_1/ConstConst*
_output_shapes
: *
dtype0*
value	B :2
while/lstm_cell_1/Const
!while/lstm_cell_1/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2#
!while/lstm_cell_1/split/split_dimÄ
&while/lstm_cell_1/split/ReadVariableOpReadVariableOp1while_lstm_cell_1_split_readvariableop_resource_0* 
_output_shapes
:
*
dtype02(
&while/lstm_cell_1/split/ReadVariableOp÷
while/lstm_cell_1/splitSplit*while/lstm_cell_1/split/split_dim:output:0.while/lstm_cell_1/split/ReadVariableOp:value:0*
T0*D
_output_shapes2
0:
:
:
:
*
	num_split2
while/lstm_cell_1/splitÅ
while/lstm_cell_1/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:0 while/lstm_cell_1/split:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
while/lstm_cell_1/MatMulÉ
while/lstm_cell_1/MatMul_1MatMul0while/TensorArrayV2Read/TensorListGetItem:item:0 while/lstm_cell_1/split:output:1*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
while/lstm_cell_1/MatMul_1É
while/lstm_cell_1/MatMul_2MatMul0while/TensorArrayV2Read/TensorListGetItem:item:0 while/lstm_cell_1/split:output:2*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
while/lstm_cell_1/MatMul_2É
while/lstm_cell_1/MatMul_3MatMul0while/TensorArrayV2Read/TensorListGetItem:item:0 while/lstm_cell_1/split:output:3*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
while/lstm_cell_1/MatMul_3x
while/lstm_cell_1/Const_1Const*
_output_shapes
: *
dtype0*
value	B :2
while/lstm_cell_1/Const_1
#while/lstm_cell_1/split_1/split_dimConst*
_output_shapes
: *
dtype0*
value	B : 2%
#while/lstm_cell_1/split_1/split_dimÅ
(while/lstm_cell_1/split_1/ReadVariableOpReadVariableOp3while_lstm_cell_1_split_1_readvariableop_resource_0*
_output_shapes	
:*
dtype02*
(while/lstm_cell_1/split_1/ReadVariableOpë
while/lstm_cell_1/split_1Split,while/lstm_cell_1/split_1/split_dim:output:00while/lstm_cell_1/split_1/ReadVariableOp:value:0*
T0*0
_output_shapes
::::*
	num_split2
while/lstm_cell_1/split_1¼
while/lstm_cell_1/BiasAddBiasAdd"while/lstm_cell_1/MatMul:product:0"while/lstm_cell_1/split_1:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
while/lstm_cell_1/BiasAddÂ
while/lstm_cell_1/BiasAdd_1BiasAdd$while/lstm_cell_1/MatMul_1:product:0"while/lstm_cell_1/split_1:output:1*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
while/lstm_cell_1/BiasAdd_1Â
while/lstm_cell_1/BiasAdd_2BiasAdd$while/lstm_cell_1/MatMul_2:product:0"while/lstm_cell_1/split_1:output:2*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
while/lstm_cell_1/BiasAdd_2Â
while/lstm_cell_1/BiasAdd_3BiasAdd$while/lstm_cell_1/MatMul_3:product:0"while/lstm_cell_1/split_1:output:3*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
while/lstm_cell_1/BiasAdd_3¢
while/lstm_cell_1/mulMulwhile_placeholder_3#while/lstm_cell_1/dropout/Mul_1:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
while/lstm_cell_1/mul¨
while/lstm_cell_1/mul_1Mulwhile_placeholder_3%while/lstm_cell_1/dropout_1/Mul_1:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
while/lstm_cell_1/mul_1¨
while/lstm_cell_1/mul_2Mulwhile_placeholder_3%while/lstm_cell_1/dropout_2/Mul_1:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
while/lstm_cell_1/mul_2¨
while/lstm_cell_1/mul_3Mulwhile_placeholder_3%while/lstm_cell_1/dropout_3/Mul_1:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
while/lstm_cell_1/mul_3²
 while/lstm_cell_1/ReadVariableOpReadVariableOp+while_lstm_cell_1_readvariableop_resource_0* 
_output_shapes
:
*
dtype02"
 while/lstm_cell_1/ReadVariableOp
%while/lstm_cell_1/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB"        2'
%while/lstm_cell_1/strided_slice/stack£
'while/lstm_cell_1/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       2)
'while/lstm_cell_1/strided_slice/stack_1£
'while/lstm_cell_1/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2)
'while/lstm_cell_1/strided_slice/stack_2ê
while/lstm_cell_1/strided_sliceStridedSlice(while/lstm_cell_1/ReadVariableOp:value:0.while/lstm_cell_1/strided_slice/stack:output:00while/lstm_cell_1/strided_slice/stack_1:output:00while/lstm_cell_1/strided_slice/stack_2:output:0*
Index0*
T0* 
_output_shapes
:
*

begin_mask*
end_mask2!
while/lstm_cell_1/strided_sliceº
while/lstm_cell_1/MatMul_4MatMulwhile/lstm_cell_1/mul:z:0(while/lstm_cell_1/strided_slice:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
while/lstm_cell_1/MatMul_4´
while/lstm_cell_1/addAddV2"while/lstm_cell_1/BiasAdd:output:0$while/lstm_cell_1/MatMul_4:product:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
while/lstm_cell_1/add
while/lstm_cell_1/SigmoidSigmoidwhile/lstm_cell_1/add:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
while/lstm_cell_1/Sigmoid¶
"while/lstm_cell_1/ReadVariableOp_1ReadVariableOp+while_lstm_cell_1_readvariableop_resource_0* 
_output_shapes
:
*
dtype02$
"while/lstm_cell_1/ReadVariableOp_1£
'while/lstm_cell_1/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB"       2)
'while/lstm_cell_1/strided_slice_1/stack§
)while/lstm_cell_1/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       2+
)while/lstm_cell_1/strided_slice_1/stack_1§
)while/lstm_cell_1/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2+
)while/lstm_cell_1/strided_slice_1/stack_2ö
!while/lstm_cell_1/strided_slice_1StridedSlice*while/lstm_cell_1/ReadVariableOp_1:value:00while/lstm_cell_1/strided_slice_1/stack:output:02while/lstm_cell_1/strided_slice_1/stack_1:output:02while/lstm_cell_1/strided_slice_1/stack_2:output:0*
Index0*
T0* 
_output_shapes
:
*

begin_mask*
end_mask2#
!while/lstm_cell_1/strided_slice_1¾
while/lstm_cell_1/MatMul_5MatMulwhile/lstm_cell_1/mul_1:z:0*while/lstm_cell_1/strided_slice_1:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
while/lstm_cell_1/MatMul_5º
while/lstm_cell_1/add_1AddV2$while/lstm_cell_1/BiasAdd_1:output:0$while/lstm_cell_1/MatMul_5:product:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
while/lstm_cell_1/add_1
while/lstm_cell_1/Sigmoid_1Sigmoidwhile/lstm_cell_1/add_1:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
while/lstm_cell_1/Sigmoid_1¢
while/lstm_cell_1/mul_4Mulwhile/lstm_cell_1/Sigmoid_1:y:0while_placeholder_4*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
while/lstm_cell_1/mul_4¶
"while/lstm_cell_1/ReadVariableOp_2ReadVariableOp+while_lstm_cell_1_readvariableop_resource_0* 
_output_shapes
:
*
dtype02$
"while/lstm_cell_1/ReadVariableOp_2£
'while/lstm_cell_1/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB"       2)
'while/lstm_cell_1/strided_slice_2/stack§
)while/lstm_cell_1/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       2+
)while/lstm_cell_1/strided_slice_2/stack_1§
)while/lstm_cell_1/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2+
)while/lstm_cell_1/strided_slice_2/stack_2ö
!while/lstm_cell_1/strided_slice_2StridedSlice*while/lstm_cell_1/ReadVariableOp_2:value:00while/lstm_cell_1/strided_slice_2/stack:output:02while/lstm_cell_1/strided_slice_2/stack_1:output:02while/lstm_cell_1/strided_slice_2/stack_2:output:0*
Index0*
T0* 
_output_shapes
:
*

begin_mask*
end_mask2#
!while/lstm_cell_1/strided_slice_2¾
while/lstm_cell_1/MatMul_6MatMulwhile/lstm_cell_1/mul_2:z:0*while/lstm_cell_1/strided_slice_2:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
while/lstm_cell_1/MatMul_6º
while/lstm_cell_1/add_2AddV2$while/lstm_cell_1/BiasAdd_2:output:0$while/lstm_cell_1/MatMul_6:product:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
while/lstm_cell_1/add_2
while/lstm_cell_1/TanhTanhwhile/lstm_cell_1/add_2:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
while/lstm_cell_1/Tanh§
while/lstm_cell_1/mul_5Mulwhile/lstm_cell_1/Sigmoid:y:0while/lstm_cell_1/Tanh:y:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
while/lstm_cell_1/mul_5¨
while/lstm_cell_1/add_3AddV2while/lstm_cell_1/mul_4:z:0while/lstm_cell_1/mul_5:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
while/lstm_cell_1/add_3¶
"while/lstm_cell_1/ReadVariableOp_3ReadVariableOp+while_lstm_cell_1_readvariableop_resource_0* 
_output_shapes
:
*
dtype02$
"while/lstm_cell_1/ReadVariableOp_3£
'while/lstm_cell_1/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB"       2)
'while/lstm_cell_1/strided_slice_3/stack§
)while/lstm_cell_1/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        2+
)while/lstm_cell_1/strided_slice_3/stack_1§
)while/lstm_cell_1/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2+
)while/lstm_cell_1/strided_slice_3/stack_2ö
!while/lstm_cell_1/strided_slice_3StridedSlice*while/lstm_cell_1/ReadVariableOp_3:value:00while/lstm_cell_1/strided_slice_3/stack:output:02while/lstm_cell_1/strided_slice_3/stack_1:output:02while/lstm_cell_1/strided_slice_3/stack_2:output:0*
Index0*
T0* 
_output_shapes
:
*

begin_mask*
end_mask2#
!while/lstm_cell_1/strided_slice_3¾
while/lstm_cell_1/MatMul_7MatMulwhile/lstm_cell_1/mul_3:z:0*while/lstm_cell_1/strided_slice_3:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
while/lstm_cell_1/MatMul_7º
while/lstm_cell_1/add_4AddV2$while/lstm_cell_1/BiasAdd_3:output:0$while/lstm_cell_1/MatMul_7:product:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
while/lstm_cell_1/add_4
while/lstm_cell_1/Sigmoid_2Sigmoidwhile/lstm_cell_1/add_4:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
while/lstm_cell_1/Sigmoid_2
while/lstm_cell_1/Tanh_1Tanhwhile/lstm_cell_1/add_3:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
while/lstm_cell_1/Tanh_1«
while/lstm_cell_1/mul_6Mulwhile/lstm_cell_1/Sigmoid_2:y:0while/lstm_cell_1/Tanh_1:y:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
while/lstm_cell_1/mul_6}
while/Tile/multiplesConst*
_output_shapes
:*
dtype0*
valueB"      2
while/Tile/multiples¥

while/TileTile2while/TensorArrayV2Read_1/TensorListGetItem:item:0while/Tile/multiples:output:0*
T0
*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

while/Tile¦
while/SelectV2SelectV2while/Tile:output:0while/lstm_cell_1/mul_6:z:0while_placeholder_2*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
while/SelectV2
while/Tile_1/multiplesConst*
_output_shapes
:*
dtype0*
valueB"      2
while/Tile_1/multiples«
while/Tile_1Tile2while/TensorArrayV2Read_1/TensorListGetItem:item:0while/Tile_1/multiples:output:0*
T0
*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
while/Tile_1
while/Tile_2/multiplesConst*
_output_shapes
:*
dtype0*
valueB"      2
while/Tile_2/multiples«
while/Tile_2Tile2while/TensorArrayV2Read_1/TensorListGetItem:item:0while/Tile_2/multiples:output:0*
T0
*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
while/Tile_2¬
while/SelectV2_1SelectV2while/Tile_1:output:0while/lstm_cell_1/mul_6:z:0while_placeholder_3*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
while/SelectV2_1¬
while/SelectV2_2SelectV2while/Tile_2:output:0while/lstm_cell_1/add_3:z:0while_placeholder_4*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
while/SelectV2_2Û
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholderwhile/SelectV2:output:0*
_output_shapes
: *
element_dtype02,
*while/TensorArrayV2Write/TensorListSetItem\
while/add/yConst*
_output_shapes
: *
dtype0*
value	B :2
while/add/yi
	while/addAddV2while_placeholderwhile/add/y:output:0*
T0*
_output_shapes
: 2
	while/add`
while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :2
while/add_1/yv
while/add_1AddV2while_while_loop_counterwhile/add_1/y:output:0*
T0*
_output_shapes
: 2
while/add_1Ä
while/IdentityIdentitywhile/add_1:z:0!^while/lstm_cell_1/ReadVariableOp#^while/lstm_cell_1/ReadVariableOp_1#^while/lstm_cell_1/ReadVariableOp_2#^while/lstm_cell_1/ReadVariableOp_3'^while/lstm_cell_1/split/ReadVariableOp)^while/lstm_cell_1/split_1/ReadVariableOp*
T0*
_output_shapes
: 2
while/Identity×
while/Identity_1Identitywhile_while_maximum_iterations!^while/lstm_cell_1/ReadVariableOp#^while/lstm_cell_1/ReadVariableOp_1#^while/lstm_cell_1/ReadVariableOp_2#^while/lstm_cell_1/ReadVariableOp_3'^while/lstm_cell_1/split/ReadVariableOp)^while/lstm_cell_1/split_1/ReadVariableOp*
T0*
_output_shapes
: 2
while/Identity_1Æ
while/Identity_2Identitywhile/add:z:0!^while/lstm_cell_1/ReadVariableOp#^while/lstm_cell_1/ReadVariableOp_1#^while/lstm_cell_1/ReadVariableOp_2#^while/lstm_cell_1/ReadVariableOp_3'^while/lstm_cell_1/split/ReadVariableOp)^while/lstm_cell_1/split_1/ReadVariableOp*
T0*
_output_shapes
: 2
while/Identity_2ó
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0!^while/lstm_cell_1/ReadVariableOp#^while/lstm_cell_1/ReadVariableOp_1#^while/lstm_cell_1/ReadVariableOp_2#^while/lstm_cell_1/ReadVariableOp_3'^while/lstm_cell_1/split/ReadVariableOp)^while/lstm_cell_1/split_1/ReadVariableOp*
T0*
_output_shapes
: 2
while/Identity_3â
while/Identity_4Identitywhile/SelectV2:output:0!^while/lstm_cell_1/ReadVariableOp#^while/lstm_cell_1/ReadVariableOp_1#^while/lstm_cell_1/ReadVariableOp_2#^while/lstm_cell_1/ReadVariableOp_3'^while/lstm_cell_1/split/ReadVariableOp)^while/lstm_cell_1/split_1/ReadVariableOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
while/Identity_4ä
while/Identity_5Identitywhile/SelectV2_1:output:0!^while/lstm_cell_1/ReadVariableOp#^while/lstm_cell_1/ReadVariableOp_1#^while/lstm_cell_1/ReadVariableOp_2#^while/lstm_cell_1/ReadVariableOp_3'^while/lstm_cell_1/split/ReadVariableOp)^while/lstm_cell_1/split_1/ReadVariableOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
while/Identity_5ä
while/Identity_6Identitywhile/SelectV2_2:output:0!^while/lstm_cell_1/ReadVariableOp#^while/lstm_cell_1/ReadVariableOp_1#^while/lstm_cell_1/ReadVariableOp_2#^while/lstm_cell_1/ReadVariableOp_3'^while/lstm_cell_1/split/ReadVariableOp)^while/lstm_cell_1/split_1/ReadVariableOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
while/Identity_6")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"-
while_identity_5while/Identity_5:output:0"-
while_identity_6while/Identity_6:output:0"X
)while_lstm_cell_1_readvariableop_resource+while_lstm_cell_1_readvariableop_resource_0"h
1while_lstm_cell_1_split_1_readvariableop_resource3while_lstm_cell_1_split_1_readvariableop_resource_0"d
/while_lstm_cell_1_split_readvariableop_resource1while_lstm_cell_1_split_readvariableop_resource_0",
while_strided_slicewhile_strided_slice_0"°
Uwhile_tensorarrayv2read_1_tensorlistgetitem_tensorarrayunstack_1_tensorlistfromtensorWwhile_tensorarrayv2read_1_tensorlistgetitem_tensorarrayunstack_1_tensorlistfromtensor_0"¨
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*i
_input_shapesX
V: : : : :ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ: : : :::2D
 while/lstm_cell_1/ReadVariableOp while/lstm_cell_1/ReadVariableOp2H
"while/lstm_cell_1/ReadVariableOp_1"while/lstm_cell_1/ReadVariableOp_12H
"while/lstm_cell_1/ReadVariableOp_2"while/lstm_cell_1/ReadVariableOp_22H
"while/lstm_cell_1/ReadVariableOp_3"while/lstm_cell_1/ReadVariableOp_32P
&while/lstm_cell_1/split/ReadVariableOp&while/lstm_cell_1/split/ReadVariableOp2T
(while/lstm_cell_1/split_1/ReadVariableOp(while/lstm_cell_1/split_1/ReadVariableOp: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :.*
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ:.*
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ:.*
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ:

_output_shapes
: :

_output_shapes
: :	

_output_shapes
: 
ÿ
b
D__inference_dropout_1_layer_call_and_return_conditional_losses_14471

inputs

identity_1h
IdentityIdentityinputs*
T0*5
_output_shapes#
!:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2

Identityw

Identity_1IdentityIdentity:output:0*
T0*5
_output_shapes#
!:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2

Identity_1"!

identity_1Identity_1:output:0*4
_input_shapes#
!:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ:] Y
5
_output_shapes#
!:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
Ì
Õ
A__inference_lstm_1_layer_call_and_return_conditional_losses_16621
inputs_0-
)lstm_cell_1_split_readvariableop_resource/
+lstm_cell_1_split_1_readvariableop_resource'
#lstm_cell_1_readvariableop_resource
identity

identity_1

identity_2¢lstm_cell_1/ReadVariableOp¢lstm_cell_1/ReadVariableOp_1¢lstm_cell_1/ReadVariableOp_2¢lstm_cell_1/ReadVariableOp_3¢ lstm_cell_1/split/ReadVariableOp¢"lstm_cell_1/split_1/ReadVariableOp¢whileF
ShapeShapeinputs_0*
T0*
_output_shapes
:2
Shapet
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice/stackx
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_1x
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_2â
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice]
zeros/mul/yConst*
_output_shapes
: *
dtype0*
value
B :2
zeros/mul/yl
	zeros/mulMulstrided_slice:output:0zeros/mul/y:output:0*
T0*
_output_shapes
: 2
	zeros/mul_
zeros/Less/yConst*
_output_shapes
: *
dtype0*
value
B :è2
zeros/Less/yg

zeros/LessLesszeros/mul:z:0zeros/Less/y:output:0*
T0*
_output_shapes
: 2

zeros/Lessc
zeros/packed/1Const*
_output_shapes
: *
dtype0*
value
B :2
zeros/packed/1
zeros/packedPackstrided_slice:output:0zeros/packed/1:output:0*
N*
T0*
_output_shapes
:2
zeros/packed_
zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
zeros/Constv
zerosFillzeros/packed:output:0zeros/Const:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
zerosa
zeros_1/mul/yConst*
_output_shapes
: *
dtype0*
value
B :2
zeros_1/mul/yr
zeros_1/mulMulstrided_slice:output:0zeros_1/mul/y:output:0*
T0*
_output_shapes
: 2
zeros_1/mulc
zeros_1/Less/yConst*
_output_shapes
: *
dtype0*
value
B :è2
zeros_1/Less/yo
zeros_1/LessLesszeros_1/mul:z:0zeros_1/Less/y:output:0*
T0*
_output_shapes
: 2
zeros_1/Lessg
zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value
B :2
zeros_1/packed/1
zeros_1/packedPackstrided_slice:output:0zeros_1/packed/1:output:0*
N*
T0*
_output_shapes
:2
zeros_1/packedc
zeros_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
zeros_1/Const~
zeros_1Fillzeros_1/packed:output:0zeros_1/Const:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2	
zeros_1u
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
transpose/perm
	transpose	Transposeinputs_0transpose/perm:output:0*
T0*5
_output_shapes#
!:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2
	transposeO
Shape_1Shapetranspose:y:0*
T0*
_output_shapes
:2	
Shape_1x
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_1/stack|
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_1|
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_2î
strided_slice_1StridedSliceShape_1:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice_1
TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ2
TensorArrayV2/element_shape²
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
TensorArrayV2¿
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ   27
5TensorArrayUnstack/TensorListFromTensor/element_shapeø
'TensorArrayUnstack/TensorListFromTensorTensorListFromTensortranspose:y:0>TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type02)
'TensorArrayUnstack/TensorListFromTensorx
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_2/stack|
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_2/stack_1|
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_2/stack_2ý
strided_slice_2StridedSlicetranspose:y:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
shrink_axis_mask2
strided_slice_2x
lstm_cell_1/ones_like/ShapeShapezeros:output:0*
T0*
_output_shapes
:2
lstm_cell_1/ones_like/Shape
lstm_cell_1/ones_like/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ?2
lstm_cell_1/ones_like/Constµ
lstm_cell_1/ones_likeFill$lstm_cell_1/ones_like/Shape:output:0$lstm_cell_1/ones_like/Const:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
lstm_cell_1/ones_likeh
lstm_cell_1/ConstConst*
_output_shapes
: *
dtype0*
value	B :2
lstm_cell_1/Const|
lstm_cell_1/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2
lstm_cell_1/split/split_dim°
 lstm_cell_1/split/ReadVariableOpReadVariableOp)lstm_cell_1_split_readvariableop_resource* 
_output_shapes
:
*
dtype02"
 lstm_cell_1/split/ReadVariableOpß
lstm_cell_1/splitSplit$lstm_cell_1/split/split_dim:output:0(lstm_cell_1/split/ReadVariableOp:value:0*
T0*D
_output_shapes2
0:
:
:
:
*
	num_split2
lstm_cell_1/split
lstm_cell_1/MatMulMatMulstrided_slice_2:output:0lstm_cell_1/split:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
lstm_cell_1/MatMul
lstm_cell_1/MatMul_1MatMulstrided_slice_2:output:0lstm_cell_1/split:output:1*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
lstm_cell_1/MatMul_1
lstm_cell_1/MatMul_2MatMulstrided_slice_2:output:0lstm_cell_1/split:output:2*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
lstm_cell_1/MatMul_2
lstm_cell_1/MatMul_3MatMulstrided_slice_2:output:0lstm_cell_1/split:output:3*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
lstm_cell_1/MatMul_3l
lstm_cell_1/Const_1Const*
_output_shapes
: *
dtype0*
value	B :2
lstm_cell_1/Const_1
lstm_cell_1/split_1/split_dimConst*
_output_shapes
: *
dtype0*
value	B : 2
lstm_cell_1/split_1/split_dim±
"lstm_cell_1/split_1/ReadVariableOpReadVariableOp+lstm_cell_1_split_1_readvariableop_resource*
_output_shapes	
:*
dtype02$
"lstm_cell_1/split_1/ReadVariableOpÓ
lstm_cell_1/split_1Split&lstm_cell_1/split_1/split_dim:output:0*lstm_cell_1/split_1/ReadVariableOp:value:0*
T0*0
_output_shapes
::::*
	num_split2
lstm_cell_1/split_1¤
lstm_cell_1/BiasAddBiasAddlstm_cell_1/MatMul:product:0lstm_cell_1/split_1:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
lstm_cell_1/BiasAddª
lstm_cell_1/BiasAdd_1BiasAddlstm_cell_1/MatMul_1:product:0lstm_cell_1/split_1:output:1*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
lstm_cell_1/BiasAdd_1ª
lstm_cell_1/BiasAdd_2BiasAddlstm_cell_1/MatMul_2:product:0lstm_cell_1/split_1:output:2*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
lstm_cell_1/BiasAdd_2ª
lstm_cell_1/BiasAdd_3BiasAddlstm_cell_1/MatMul_3:product:0lstm_cell_1/split_1:output:3*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
lstm_cell_1/BiasAdd_3
lstm_cell_1/mulMulzeros:output:0lstm_cell_1/ones_like:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
lstm_cell_1/mul
lstm_cell_1/mul_1Mulzeros:output:0lstm_cell_1/ones_like:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
lstm_cell_1/mul_1
lstm_cell_1/mul_2Mulzeros:output:0lstm_cell_1/ones_like:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
lstm_cell_1/mul_2
lstm_cell_1/mul_3Mulzeros:output:0lstm_cell_1/ones_like:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
lstm_cell_1/mul_3
lstm_cell_1/ReadVariableOpReadVariableOp#lstm_cell_1_readvariableop_resource* 
_output_shapes
:
*
dtype02
lstm_cell_1/ReadVariableOp
lstm_cell_1/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB"        2!
lstm_cell_1/strided_slice/stack
!lstm_cell_1/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       2#
!lstm_cell_1/strided_slice/stack_1
!lstm_cell_1/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2#
!lstm_cell_1/strided_slice/stack_2Æ
lstm_cell_1/strided_sliceStridedSlice"lstm_cell_1/ReadVariableOp:value:0(lstm_cell_1/strided_slice/stack:output:0*lstm_cell_1/strided_slice/stack_1:output:0*lstm_cell_1/strided_slice/stack_2:output:0*
Index0*
T0* 
_output_shapes
:
*

begin_mask*
end_mask2
lstm_cell_1/strided_slice¢
lstm_cell_1/MatMul_4MatMullstm_cell_1/mul:z:0"lstm_cell_1/strided_slice:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
lstm_cell_1/MatMul_4
lstm_cell_1/addAddV2lstm_cell_1/BiasAdd:output:0lstm_cell_1/MatMul_4:product:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
lstm_cell_1/add}
lstm_cell_1/SigmoidSigmoidlstm_cell_1/add:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
lstm_cell_1/Sigmoid¢
lstm_cell_1/ReadVariableOp_1ReadVariableOp#lstm_cell_1_readvariableop_resource* 
_output_shapes
:
*
dtype02
lstm_cell_1/ReadVariableOp_1
!lstm_cell_1/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB"       2#
!lstm_cell_1/strided_slice_1/stack
#lstm_cell_1/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       2%
#lstm_cell_1/strided_slice_1/stack_1
#lstm_cell_1/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2%
#lstm_cell_1/strided_slice_1/stack_2Ò
lstm_cell_1/strided_slice_1StridedSlice$lstm_cell_1/ReadVariableOp_1:value:0*lstm_cell_1/strided_slice_1/stack:output:0,lstm_cell_1/strided_slice_1/stack_1:output:0,lstm_cell_1/strided_slice_1/stack_2:output:0*
Index0*
T0* 
_output_shapes
:
*

begin_mask*
end_mask2
lstm_cell_1/strided_slice_1¦
lstm_cell_1/MatMul_5MatMullstm_cell_1/mul_1:z:0$lstm_cell_1/strided_slice_1:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
lstm_cell_1/MatMul_5¢
lstm_cell_1/add_1AddV2lstm_cell_1/BiasAdd_1:output:0lstm_cell_1/MatMul_5:product:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
lstm_cell_1/add_1
lstm_cell_1/Sigmoid_1Sigmoidlstm_cell_1/add_1:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
lstm_cell_1/Sigmoid_1
lstm_cell_1/mul_4Mullstm_cell_1/Sigmoid_1:y:0zeros_1:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
lstm_cell_1/mul_4¢
lstm_cell_1/ReadVariableOp_2ReadVariableOp#lstm_cell_1_readvariableop_resource* 
_output_shapes
:
*
dtype02
lstm_cell_1/ReadVariableOp_2
!lstm_cell_1/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB"       2#
!lstm_cell_1/strided_slice_2/stack
#lstm_cell_1/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       2%
#lstm_cell_1/strided_slice_2/stack_1
#lstm_cell_1/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2%
#lstm_cell_1/strided_slice_2/stack_2Ò
lstm_cell_1/strided_slice_2StridedSlice$lstm_cell_1/ReadVariableOp_2:value:0*lstm_cell_1/strided_slice_2/stack:output:0,lstm_cell_1/strided_slice_2/stack_1:output:0,lstm_cell_1/strided_slice_2/stack_2:output:0*
Index0*
T0* 
_output_shapes
:
*

begin_mask*
end_mask2
lstm_cell_1/strided_slice_2¦
lstm_cell_1/MatMul_6MatMullstm_cell_1/mul_2:z:0$lstm_cell_1/strided_slice_2:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
lstm_cell_1/MatMul_6¢
lstm_cell_1/add_2AddV2lstm_cell_1/BiasAdd_2:output:0lstm_cell_1/MatMul_6:product:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
lstm_cell_1/add_2v
lstm_cell_1/TanhTanhlstm_cell_1/add_2:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
lstm_cell_1/Tanh
lstm_cell_1/mul_5Mullstm_cell_1/Sigmoid:y:0lstm_cell_1/Tanh:y:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
lstm_cell_1/mul_5
lstm_cell_1/add_3AddV2lstm_cell_1/mul_4:z:0lstm_cell_1/mul_5:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
lstm_cell_1/add_3¢
lstm_cell_1/ReadVariableOp_3ReadVariableOp#lstm_cell_1_readvariableop_resource* 
_output_shapes
:
*
dtype02
lstm_cell_1/ReadVariableOp_3
!lstm_cell_1/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB"       2#
!lstm_cell_1/strided_slice_3/stack
#lstm_cell_1/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        2%
#lstm_cell_1/strided_slice_3/stack_1
#lstm_cell_1/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2%
#lstm_cell_1/strided_slice_3/stack_2Ò
lstm_cell_1/strided_slice_3StridedSlice$lstm_cell_1/ReadVariableOp_3:value:0*lstm_cell_1/strided_slice_3/stack:output:0,lstm_cell_1/strided_slice_3/stack_1:output:0,lstm_cell_1/strided_slice_3/stack_2:output:0*
Index0*
T0* 
_output_shapes
:
*

begin_mask*
end_mask2
lstm_cell_1/strided_slice_3¦
lstm_cell_1/MatMul_7MatMullstm_cell_1/mul_3:z:0$lstm_cell_1/strided_slice_3:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
lstm_cell_1/MatMul_7¢
lstm_cell_1/add_4AddV2lstm_cell_1/BiasAdd_3:output:0lstm_cell_1/MatMul_7:product:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
lstm_cell_1/add_4
lstm_cell_1/Sigmoid_2Sigmoidlstm_cell_1/add_4:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
lstm_cell_1/Sigmoid_2z
lstm_cell_1/Tanh_1Tanhlstm_cell_1/add_3:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
lstm_cell_1/Tanh_1
lstm_cell_1/mul_6Mullstm_cell_1/Sigmoid_2:y:0lstm_cell_1/Tanh_1:y:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
lstm_cell_1/mul_6
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ   2
TensorArrayV2_1/element_shape¸
TensorArrayV2_1TensorListReserve&TensorArrayV2_1/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
TensorArrayV2_1N
timeConst*
_output_shapes
: *
dtype0*
value	B : 2
time
while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ2
while/maximum_iterationsj
while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : 2
while/loop_counterã
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0zeros_1:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0)lstm_cell_1_split_readvariableop_resource+lstm_cell_1_split_1_readvariableop_resource#lstm_cell_1_readvariableop_resource*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*N
_output_shapes<
:: : : : :ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ: : : : : *%
_read_only_resource_inputs
	
*
bodyR
while_body_16490*
condR
while_cond_16489*M
output_shapes<
:: : : : :ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ: : : : : *
parallel_iterations 2
whileµ
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ   22
0TensorArrayV2Stack/TensorListStack/element_shapeò
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*5
_output_shapes#
!:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ*
element_dtype02$
"TensorArrayV2Stack/TensorListStack
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
ÿÿÿÿÿÿÿÿÿ2
strided_slice_3/stack|
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_3/stack_1|
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_3/stack_2
strided_slice_3StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
shrink_axis_mask2
strided_slice_3y
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
transpose_1/perm¯
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*5
_output_shapes#
!:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2
transpose_1f
runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    2	
runtime»
IdentityIdentitytranspose_1:y:0^lstm_cell_1/ReadVariableOp^lstm_cell_1/ReadVariableOp_1^lstm_cell_1/ReadVariableOp_2^lstm_cell_1/ReadVariableOp_3!^lstm_cell_1/split/ReadVariableOp#^lstm_cell_1/split_1/ReadVariableOp^while*
T0*5
_output_shapes#
!:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2

Identity±

Identity_1Identitywhile:output:4^lstm_cell_1/ReadVariableOp^lstm_cell_1/ReadVariableOp_1^lstm_cell_1/ReadVariableOp_2^lstm_cell_1/ReadVariableOp_3!^lstm_cell_1/split/ReadVariableOp#^lstm_cell_1/split_1/ReadVariableOp^while*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity_1±

Identity_2Identitywhile:output:5^lstm_cell_1/ReadVariableOp^lstm_cell_1/ReadVariableOp_1^lstm_cell_1/ReadVariableOp_2^lstm_cell_1/ReadVariableOp_3!^lstm_cell_1/split/ReadVariableOp#^lstm_cell_1/split_1/ReadVariableOp^while*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity_2"
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0*@
_input_shapes/
-:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ:::28
lstm_cell_1/ReadVariableOplstm_cell_1/ReadVariableOp2<
lstm_cell_1/ReadVariableOp_1lstm_cell_1/ReadVariableOp_12<
lstm_cell_1/ReadVariableOp_2lstm_cell_1/ReadVariableOp_22<
lstm_cell_1/ReadVariableOp_3lstm_cell_1/ReadVariableOp_32D
 lstm_cell_1/split/ReadVariableOp lstm_cell_1/split/ReadVariableOp2H
"lstm_cell_1/split_1/ReadVariableOp"lstm_cell_1/split_1/ReadVariableOp2
whilewhile:_ [
5
_output_shapes#
!:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
"
_user_specified_name
inputs/0
î´
Ç
lstm_1_while_body_15789*
&lstm_1_while_lstm_1_while_loop_counter0
,lstm_1_while_lstm_1_while_maximum_iterations
lstm_1_while_placeholder
lstm_1_while_placeholder_1
lstm_1_while_placeholder_2
lstm_1_while_placeholder_3
lstm_1_while_placeholder_4'
#lstm_1_while_lstm_1_strided_slice_0e
alstm_1_while_tensorarrayv2read_tensorlistgetitem_lstm_1_tensorarrayunstack_tensorlistfromtensor_0i
elstm_1_while_tensorarrayv2read_1_tensorlistgetitem_lstm_1_tensorarrayunstack_1_tensorlistfromtensor_0<
8lstm_1_while_lstm_cell_1_split_readvariableop_resource_0>
:lstm_1_while_lstm_cell_1_split_1_readvariableop_resource_06
2lstm_1_while_lstm_cell_1_readvariableop_resource_0
lstm_1_while_identity
lstm_1_while_identity_1
lstm_1_while_identity_2
lstm_1_while_identity_3
lstm_1_while_identity_4
lstm_1_while_identity_5
lstm_1_while_identity_6%
!lstm_1_while_lstm_1_strided_slicec
_lstm_1_while_tensorarrayv2read_tensorlistgetitem_lstm_1_tensorarrayunstack_tensorlistfromtensorg
clstm_1_while_tensorarrayv2read_1_tensorlistgetitem_lstm_1_tensorarrayunstack_1_tensorlistfromtensor:
6lstm_1_while_lstm_cell_1_split_readvariableop_resource<
8lstm_1_while_lstm_cell_1_split_1_readvariableop_resource4
0lstm_1_while_lstm_cell_1_readvariableop_resource¢'lstm_1/while/lstm_cell_1/ReadVariableOp¢)lstm_1/while/lstm_cell_1/ReadVariableOp_1¢)lstm_1/while/lstm_cell_1/ReadVariableOp_2¢)lstm_1/while/lstm_cell_1/ReadVariableOp_3¢-lstm_1/while/lstm_cell_1/split/ReadVariableOp¢/lstm_1/while/lstm_cell_1/split_1/ReadVariableOpÑ
>lstm_1/while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ   2@
>lstm_1/while/TensorArrayV2Read/TensorListGetItem/element_shapeþ
0lstm_1/while/TensorArrayV2Read/TensorListGetItemTensorListGetItemalstm_1_while_tensorarrayv2read_tensorlistgetitem_lstm_1_tensorarrayunstack_tensorlistfromtensor_0lstm_1_while_placeholderGlstm_1/while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
element_dtype022
0lstm_1/while/TensorArrayV2Read/TensorListGetItemÕ
@lstm_1/while/TensorArrayV2Read_1/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ   2B
@lstm_1/while/TensorArrayV2Read_1/TensorListGetItem/element_shape
2lstm_1/while/TensorArrayV2Read_1/TensorListGetItemTensorListGetItemelstm_1_while_tensorarrayv2read_1_tensorlistgetitem_lstm_1_tensorarrayunstack_1_tensorlistfromtensor_0lstm_1_while_placeholderIlstm_1/while/TensorArrayV2Read_1/TensorListGetItem/element_shape:output:0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
element_dtype0
24
2lstm_1/while/TensorArrayV2Read_1/TensorListGetItem
(lstm_1/while/lstm_cell_1/ones_like/ShapeShapelstm_1_while_placeholder_3*
T0*
_output_shapes
:2*
(lstm_1/while/lstm_cell_1/ones_like/Shape
(lstm_1/while/lstm_cell_1/ones_like/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ?2*
(lstm_1/while/lstm_cell_1/ones_like/Consté
"lstm_1/while/lstm_cell_1/ones_likeFill1lstm_1/while/lstm_cell_1/ones_like/Shape:output:01lstm_1/while/lstm_cell_1/ones_like/Const:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2$
"lstm_1/while/lstm_cell_1/ones_like
lstm_1/while/lstm_cell_1/ConstConst*
_output_shapes
: *
dtype0*
value	B :2 
lstm_1/while/lstm_cell_1/Const
(lstm_1/while/lstm_cell_1/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2*
(lstm_1/while/lstm_cell_1/split/split_dimÙ
-lstm_1/while/lstm_cell_1/split/ReadVariableOpReadVariableOp8lstm_1_while_lstm_cell_1_split_readvariableop_resource_0* 
_output_shapes
:
*
dtype02/
-lstm_1/while/lstm_cell_1/split/ReadVariableOp
lstm_1/while/lstm_cell_1/splitSplit1lstm_1/while/lstm_cell_1/split/split_dim:output:05lstm_1/while/lstm_cell_1/split/ReadVariableOp:value:0*
T0*D
_output_shapes2
0:
:
:
:
*
	num_split2 
lstm_1/while/lstm_cell_1/splitá
lstm_1/while/lstm_cell_1/MatMulMatMul7lstm_1/while/TensorArrayV2Read/TensorListGetItem:item:0'lstm_1/while/lstm_cell_1/split:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2!
lstm_1/while/lstm_cell_1/MatMulå
!lstm_1/while/lstm_cell_1/MatMul_1MatMul7lstm_1/while/TensorArrayV2Read/TensorListGetItem:item:0'lstm_1/while/lstm_cell_1/split:output:1*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2#
!lstm_1/while/lstm_cell_1/MatMul_1å
!lstm_1/while/lstm_cell_1/MatMul_2MatMul7lstm_1/while/TensorArrayV2Read/TensorListGetItem:item:0'lstm_1/while/lstm_cell_1/split:output:2*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2#
!lstm_1/while/lstm_cell_1/MatMul_2å
!lstm_1/while/lstm_cell_1/MatMul_3MatMul7lstm_1/while/TensorArrayV2Read/TensorListGetItem:item:0'lstm_1/while/lstm_cell_1/split:output:3*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2#
!lstm_1/while/lstm_cell_1/MatMul_3
 lstm_1/while/lstm_cell_1/Const_1Const*
_output_shapes
: *
dtype0*
value	B :2"
 lstm_1/while/lstm_cell_1/Const_1
*lstm_1/while/lstm_cell_1/split_1/split_dimConst*
_output_shapes
: *
dtype0*
value	B : 2,
*lstm_1/while/lstm_cell_1/split_1/split_dimÚ
/lstm_1/while/lstm_cell_1/split_1/ReadVariableOpReadVariableOp:lstm_1_while_lstm_cell_1_split_1_readvariableop_resource_0*
_output_shapes	
:*
dtype021
/lstm_1/while/lstm_cell_1/split_1/ReadVariableOp
 lstm_1/while/lstm_cell_1/split_1Split3lstm_1/while/lstm_cell_1/split_1/split_dim:output:07lstm_1/while/lstm_cell_1/split_1/ReadVariableOp:value:0*
T0*0
_output_shapes
::::*
	num_split2"
 lstm_1/while/lstm_cell_1/split_1Ø
 lstm_1/while/lstm_cell_1/BiasAddBiasAdd)lstm_1/while/lstm_cell_1/MatMul:product:0)lstm_1/while/lstm_cell_1/split_1:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2"
 lstm_1/while/lstm_cell_1/BiasAddÞ
"lstm_1/while/lstm_cell_1/BiasAdd_1BiasAdd+lstm_1/while/lstm_cell_1/MatMul_1:product:0)lstm_1/while/lstm_cell_1/split_1:output:1*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2$
"lstm_1/while/lstm_cell_1/BiasAdd_1Þ
"lstm_1/while/lstm_cell_1/BiasAdd_2BiasAdd+lstm_1/while/lstm_cell_1/MatMul_2:product:0)lstm_1/while/lstm_cell_1/split_1:output:2*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2$
"lstm_1/while/lstm_cell_1/BiasAdd_2Þ
"lstm_1/while/lstm_cell_1/BiasAdd_3BiasAdd+lstm_1/while/lstm_cell_1/MatMul_3:product:0)lstm_1/while/lstm_cell_1/split_1:output:3*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2$
"lstm_1/while/lstm_cell_1/BiasAdd_3¿
lstm_1/while/lstm_cell_1/mulMullstm_1_while_placeholder_3+lstm_1/while/lstm_cell_1/ones_like:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
lstm_1/while/lstm_cell_1/mulÃ
lstm_1/while/lstm_cell_1/mul_1Mullstm_1_while_placeholder_3+lstm_1/while/lstm_cell_1/ones_like:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2 
lstm_1/while/lstm_cell_1/mul_1Ã
lstm_1/while/lstm_cell_1/mul_2Mullstm_1_while_placeholder_3+lstm_1/while/lstm_cell_1/ones_like:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2 
lstm_1/while/lstm_cell_1/mul_2Ã
lstm_1/while/lstm_cell_1/mul_3Mullstm_1_while_placeholder_3+lstm_1/while/lstm_cell_1/ones_like:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2 
lstm_1/while/lstm_cell_1/mul_3Ç
'lstm_1/while/lstm_cell_1/ReadVariableOpReadVariableOp2lstm_1_while_lstm_cell_1_readvariableop_resource_0* 
_output_shapes
:
*
dtype02)
'lstm_1/while/lstm_cell_1/ReadVariableOp­
,lstm_1/while/lstm_cell_1/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB"        2.
,lstm_1/while/lstm_cell_1/strided_slice/stack±
.lstm_1/while/lstm_cell_1/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       20
.lstm_1/while/lstm_cell_1/strided_slice/stack_1±
.lstm_1/while/lstm_cell_1/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      20
.lstm_1/while/lstm_cell_1/strided_slice/stack_2
&lstm_1/while/lstm_cell_1/strided_sliceStridedSlice/lstm_1/while/lstm_cell_1/ReadVariableOp:value:05lstm_1/while/lstm_cell_1/strided_slice/stack:output:07lstm_1/while/lstm_cell_1/strided_slice/stack_1:output:07lstm_1/while/lstm_cell_1/strided_slice/stack_2:output:0*
Index0*
T0* 
_output_shapes
:
*

begin_mask*
end_mask2(
&lstm_1/while/lstm_cell_1/strided_sliceÖ
!lstm_1/while/lstm_cell_1/MatMul_4MatMul lstm_1/while/lstm_cell_1/mul:z:0/lstm_1/while/lstm_cell_1/strided_slice:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2#
!lstm_1/while/lstm_cell_1/MatMul_4Ð
lstm_1/while/lstm_cell_1/addAddV2)lstm_1/while/lstm_cell_1/BiasAdd:output:0+lstm_1/while/lstm_cell_1/MatMul_4:product:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
lstm_1/while/lstm_cell_1/add¤
 lstm_1/while/lstm_cell_1/SigmoidSigmoid lstm_1/while/lstm_cell_1/add:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2"
 lstm_1/while/lstm_cell_1/SigmoidË
)lstm_1/while/lstm_cell_1/ReadVariableOp_1ReadVariableOp2lstm_1_while_lstm_cell_1_readvariableop_resource_0* 
_output_shapes
:
*
dtype02+
)lstm_1/while/lstm_cell_1/ReadVariableOp_1±
.lstm_1/while/lstm_cell_1/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB"       20
.lstm_1/while/lstm_cell_1/strided_slice_1/stackµ
0lstm_1/while/lstm_cell_1/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       22
0lstm_1/while/lstm_cell_1/strided_slice_1/stack_1µ
0lstm_1/while/lstm_cell_1/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      22
0lstm_1/while/lstm_cell_1/strided_slice_1/stack_2 
(lstm_1/while/lstm_cell_1/strided_slice_1StridedSlice1lstm_1/while/lstm_cell_1/ReadVariableOp_1:value:07lstm_1/while/lstm_cell_1/strided_slice_1/stack:output:09lstm_1/while/lstm_cell_1/strided_slice_1/stack_1:output:09lstm_1/while/lstm_cell_1/strided_slice_1/stack_2:output:0*
Index0*
T0* 
_output_shapes
:
*

begin_mask*
end_mask2*
(lstm_1/while/lstm_cell_1/strided_slice_1Ú
!lstm_1/while/lstm_cell_1/MatMul_5MatMul"lstm_1/while/lstm_cell_1/mul_1:z:01lstm_1/while/lstm_cell_1/strided_slice_1:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2#
!lstm_1/while/lstm_cell_1/MatMul_5Ö
lstm_1/while/lstm_cell_1/add_1AddV2+lstm_1/while/lstm_cell_1/BiasAdd_1:output:0+lstm_1/while/lstm_cell_1/MatMul_5:product:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2 
lstm_1/while/lstm_cell_1/add_1ª
"lstm_1/while/lstm_cell_1/Sigmoid_1Sigmoid"lstm_1/while/lstm_cell_1/add_1:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2$
"lstm_1/while/lstm_cell_1/Sigmoid_1¾
lstm_1/while/lstm_cell_1/mul_4Mul&lstm_1/while/lstm_cell_1/Sigmoid_1:y:0lstm_1_while_placeholder_4*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2 
lstm_1/while/lstm_cell_1/mul_4Ë
)lstm_1/while/lstm_cell_1/ReadVariableOp_2ReadVariableOp2lstm_1_while_lstm_cell_1_readvariableop_resource_0* 
_output_shapes
:
*
dtype02+
)lstm_1/while/lstm_cell_1/ReadVariableOp_2±
.lstm_1/while/lstm_cell_1/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB"       20
.lstm_1/while/lstm_cell_1/strided_slice_2/stackµ
0lstm_1/while/lstm_cell_1/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       22
0lstm_1/while/lstm_cell_1/strided_slice_2/stack_1µ
0lstm_1/while/lstm_cell_1/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      22
0lstm_1/while/lstm_cell_1/strided_slice_2/stack_2 
(lstm_1/while/lstm_cell_1/strided_slice_2StridedSlice1lstm_1/while/lstm_cell_1/ReadVariableOp_2:value:07lstm_1/while/lstm_cell_1/strided_slice_2/stack:output:09lstm_1/while/lstm_cell_1/strided_slice_2/stack_1:output:09lstm_1/while/lstm_cell_1/strided_slice_2/stack_2:output:0*
Index0*
T0* 
_output_shapes
:
*

begin_mask*
end_mask2*
(lstm_1/while/lstm_cell_1/strided_slice_2Ú
!lstm_1/while/lstm_cell_1/MatMul_6MatMul"lstm_1/while/lstm_cell_1/mul_2:z:01lstm_1/while/lstm_cell_1/strided_slice_2:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2#
!lstm_1/while/lstm_cell_1/MatMul_6Ö
lstm_1/while/lstm_cell_1/add_2AddV2+lstm_1/while/lstm_cell_1/BiasAdd_2:output:0+lstm_1/while/lstm_cell_1/MatMul_6:product:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2 
lstm_1/while/lstm_cell_1/add_2
lstm_1/while/lstm_cell_1/TanhTanh"lstm_1/while/lstm_cell_1/add_2:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
lstm_1/while/lstm_cell_1/TanhÃ
lstm_1/while/lstm_cell_1/mul_5Mul$lstm_1/while/lstm_cell_1/Sigmoid:y:0!lstm_1/while/lstm_cell_1/Tanh:y:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2 
lstm_1/while/lstm_cell_1/mul_5Ä
lstm_1/while/lstm_cell_1/add_3AddV2"lstm_1/while/lstm_cell_1/mul_4:z:0"lstm_1/while/lstm_cell_1/mul_5:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2 
lstm_1/while/lstm_cell_1/add_3Ë
)lstm_1/while/lstm_cell_1/ReadVariableOp_3ReadVariableOp2lstm_1_while_lstm_cell_1_readvariableop_resource_0* 
_output_shapes
:
*
dtype02+
)lstm_1/while/lstm_cell_1/ReadVariableOp_3±
.lstm_1/while/lstm_cell_1/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB"       20
.lstm_1/while/lstm_cell_1/strided_slice_3/stackµ
0lstm_1/while/lstm_cell_1/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        22
0lstm_1/while/lstm_cell_1/strided_slice_3/stack_1µ
0lstm_1/while/lstm_cell_1/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      22
0lstm_1/while/lstm_cell_1/strided_slice_3/stack_2 
(lstm_1/while/lstm_cell_1/strided_slice_3StridedSlice1lstm_1/while/lstm_cell_1/ReadVariableOp_3:value:07lstm_1/while/lstm_cell_1/strided_slice_3/stack:output:09lstm_1/while/lstm_cell_1/strided_slice_3/stack_1:output:09lstm_1/while/lstm_cell_1/strided_slice_3/stack_2:output:0*
Index0*
T0* 
_output_shapes
:
*

begin_mask*
end_mask2*
(lstm_1/while/lstm_cell_1/strided_slice_3Ú
!lstm_1/while/lstm_cell_1/MatMul_7MatMul"lstm_1/while/lstm_cell_1/mul_3:z:01lstm_1/while/lstm_cell_1/strided_slice_3:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2#
!lstm_1/while/lstm_cell_1/MatMul_7Ö
lstm_1/while/lstm_cell_1/add_4AddV2+lstm_1/while/lstm_cell_1/BiasAdd_3:output:0+lstm_1/while/lstm_cell_1/MatMul_7:product:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2 
lstm_1/while/lstm_cell_1/add_4ª
"lstm_1/while/lstm_cell_1/Sigmoid_2Sigmoid"lstm_1/while/lstm_cell_1/add_4:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2$
"lstm_1/while/lstm_cell_1/Sigmoid_2¡
lstm_1/while/lstm_cell_1/Tanh_1Tanh"lstm_1/while/lstm_cell_1/add_3:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2!
lstm_1/while/lstm_cell_1/Tanh_1Ç
lstm_1/while/lstm_cell_1/mul_6Mul&lstm_1/while/lstm_cell_1/Sigmoid_2:y:0#lstm_1/while/lstm_cell_1/Tanh_1:y:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2 
lstm_1/while/lstm_cell_1/mul_6
lstm_1/while/Tile/multiplesConst*
_output_shapes
:*
dtype0*
valueB"      2
lstm_1/while/Tile/multiplesÁ
lstm_1/while/TileTile9lstm_1/while/TensorArrayV2Read_1/TensorListGetItem:item:0$lstm_1/while/Tile/multiples:output:0*
T0
*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
lstm_1/while/TileÉ
lstm_1/while/SelectV2SelectV2lstm_1/while/Tile:output:0"lstm_1/while/lstm_cell_1/mul_6:z:0lstm_1_while_placeholder_2*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
lstm_1/while/SelectV2
lstm_1/while/Tile_1/multiplesConst*
_output_shapes
:*
dtype0*
valueB"      2
lstm_1/while/Tile_1/multiplesÇ
lstm_1/while/Tile_1Tile9lstm_1/while/TensorArrayV2Read_1/TensorListGetItem:item:0&lstm_1/while/Tile_1/multiples:output:0*
T0
*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
lstm_1/while/Tile_1
lstm_1/while/Tile_2/multiplesConst*
_output_shapes
:*
dtype0*
valueB"      2
lstm_1/while/Tile_2/multiplesÇ
lstm_1/while/Tile_2Tile9lstm_1/while/TensorArrayV2Read_1/TensorListGetItem:item:0&lstm_1/while/Tile_2/multiples:output:0*
T0
*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
lstm_1/while/Tile_2Ï
lstm_1/while/SelectV2_1SelectV2lstm_1/while/Tile_1:output:0"lstm_1/while/lstm_cell_1/mul_6:z:0lstm_1_while_placeholder_3*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
lstm_1/while/SelectV2_1Ï
lstm_1/while/SelectV2_2SelectV2lstm_1/while/Tile_2:output:0"lstm_1/while/lstm_cell_1/add_3:z:0lstm_1_while_placeholder_4*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
lstm_1/while/SelectV2_2þ
1lstm_1/while/TensorArrayV2Write/TensorListSetItemTensorListSetItemlstm_1_while_placeholder_1lstm_1_while_placeholderlstm_1/while/SelectV2:output:0*
_output_shapes
: *
element_dtype023
1lstm_1/while/TensorArrayV2Write/TensorListSetItemj
lstm_1/while/add/yConst*
_output_shapes
: *
dtype0*
value	B :2
lstm_1/while/add/y
lstm_1/while/addAddV2lstm_1_while_placeholderlstm_1/while/add/y:output:0*
T0*
_output_shapes
: 2
lstm_1/while/addn
lstm_1/while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :2
lstm_1/while/add_1/y
lstm_1/while/add_1AddV2&lstm_1_while_lstm_1_while_loop_counterlstm_1/while/add_1/y:output:0*
T0*
_output_shapes
: 2
lstm_1/while/add_1
lstm_1/while/IdentityIdentitylstm_1/while/add_1:z:0(^lstm_1/while/lstm_cell_1/ReadVariableOp*^lstm_1/while/lstm_cell_1/ReadVariableOp_1*^lstm_1/while/lstm_cell_1/ReadVariableOp_2*^lstm_1/while/lstm_cell_1/ReadVariableOp_3.^lstm_1/while/lstm_cell_1/split/ReadVariableOp0^lstm_1/while/lstm_cell_1/split_1/ReadVariableOp*
T0*
_output_shapes
: 2
lstm_1/while/Identity
lstm_1/while/Identity_1Identity,lstm_1_while_lstm_1_while_maximum_iterations(^lstm_1/while/lstm_cell_1/ReadVariableOp*^lstm_1/while/lstm_cell_1/ReadVariableOp_1*^lstm_1/while/lstm_cell_1/ReadVariableOp_2*^lstm_1/while/lstm_cell_1/ReadVariableOp_3.^lstm_1/while/lstm_cell_1/split/ReadVariableOp0^lstm_1/while/lstm_cell_1/split_1/ReadVariableOp*
T0*
_output_shapes
: 2
lstm_1/while/Identity_1
lstm_1/while/Identity_2Identitylstm_1/while/add:z:0(^lstm_1/while/lstm_cell_1/ReadVariableOp*^lstm_1/while/lstm_cell_1/ReadVariableOp_1*^lstm_1/while/lstm_cell_1/ReadVariableOp_2*^lstm_1/while/lstm_cell_1/ReadVariableOp_3.^lstm_1/while/lstm_cell_1/split/ReadVariableOp0^lstm_1/while/lstm_cell_1/split_1/ReadVariableOp*
T0*
_output_shapes
: 2
lstm_1/while/Identity_2²
lstm_1/while/Identity_3IdentityAlstm_1/while/TensorArrayV2Write/TensorListSetItem:output_handle:0(^lstm_1/while/lstm_cell_1/ReadVariableOp*^lstm_1/while/lstm_cell_1/ReadVariableOp_1*^lstm_1/while/lstm_cell_1/ReadVariableOp_2*^lstm_1/while/lstm_cell_1/ReadVariableOp_3.^lstm_1/while/lstm_cell_1/split/ReadVariableOp0^lstm_1/while/lstm_cell_1/split_1/ReadVariableOp*
T0*
_output_shapes
: 2
lstm_1/while/Identity_3¡
lstm_1/while/Identity_4Identitylstm_1/while/SelectV2:output:0(^lstm_1/while/lstm_cell_1/ReadVariableOp*^lstm_1/while/lstm_cell_1/ReadVariableOp_1*^lstm_1/while/lstm_cell_1/ReadVariableOp_2*^lstm_1/while/lstm_cell_1/ReadVariableOp_3.^lstm_1/while/lstm_cell_1/split/ReadVariableOp0^lstm_1/while/lstm_cell_1/split_1/ReadVariableOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
lstm_1/while/Identity_4£
lstm_1/while/Identity_5Identity lstm_1/while/SelectV2_1:output:0(^lstm_1/while/lstm_cell_1/ReadVariableOp*^lstm_1/while/lstm_cell_1/ReadVariableOp_1*^lstm_1/while/lstm_cell_1/ReadVariableOp_2*^lstm_1/while/lstm_cell_1/ReadVariableOp_3.^lstm_1/while/lstm_cell_1/split/ReadVariableOp0^lstm_1/while/lstm_cell_1/split_1/ReadVariableOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
lstm_1/while/Identity_5£
lstm_1/while/Identity_6Identity lstm_1/while/SelectV2_2:output:0(^lstm_1/while/lstm_cell_1/ReadVariableOp*^lstm_1/while/lstm_cell_1/ReadVariableOp_1*^lstm_1/while/lstm_cell_1/ReadVariableOp_2*^lstm_1/while/lstm_cell_1/ReadVariableOp_3.^lstm_1/while/lstm_cell_1/split/ReadVariableOp0^lstm_1/while/lstm_cell_1/split_1/ReadVariableOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
lstm_1/while/Identity_6"7
lstm_1_while_identitylstm_1/while/Identity:output:0";
lstm_1_while_identity_1 lstm_1/while/Identity_1:output:0";
lstm_1_while_identity_2 lstm_1/while/Identity_2:output:0";
lstm_1_while_identity_3 lstm_1/while/Identity_3:output:0";
lstm_1_while_identity_4 lstm_1/while/Identity_4:output:0";
lstm_1_while_identity_5 lstm_1/while/Identity_5:output:0";
lstm_1_while_identity_6 lstm_1/while/Identity_6:output:0"H
!lstm_1_while_lstm_1_strided_slice#lstm_1_while_lstm_1_strided_slice_0"f
0lstm_1_while_lstm_cell_1_readvariableop_resource2lstm_1_while_lstm_cell_1_readvariableop_resource_0"v
8lstm_1_while_lstm_cell_1_split_1_readvariableop_resource:lstm_1_while_lstm_cell_1_split_1_readvariableop_resource_0"r
6lstm_1_while_lstm_cell_1_split_readvariableop_resource8lstm_1_while_lstm_cell_1_split_readvariableop_resource_0"Ì
clstm_1_while_tensorarrayv2read_1_tensorlistgetitem_lstm_1_tensorarrayunstack_1_tensorlistfromtensorelstm_1_while_tensorarrayv2read_1_tensorlistgetitem_lstm_1_tensorarrayunstack_1_tensorlistfromtensor_0"Ä
_lstm_1_while_tensorarrayv2read_tensorlistgetitem_lstm_1_tensorarrayunstack_tensorlistfromtensoralstm_1_while_tensorarrayv2read_tensorlistgetitem_lstm_1_tensorarrayunstack_tensorlistfromtensor_0*i
_input_shapesX
V: : : : :ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ: : : :::2R
'lstm_1/while/lstm_cell_1/ReadVariableOp'lstm_1/while/lstm_cell_1/ReadVariableOp2V
)lstm_1/while/lstm_cell_1/ReadVariableOp_1)lstm_1/while/lstm_cell_1/ReadVariableOp_12V
)lstm_1/while/lstm_cell_1/ReadVariableOp_2)lstm_1/while/lstm_cell_1/ReadVariableOp_22V
)lstm_1/while/lstm_cell_1/ReadVariableOp_3)lstm_1/while/lstm_cell_1/ReadVariableOp_32^
-lstm_1/while/lstm_cell_1/split/ReadVariableOp-lstm_1/while/lstm_cell_1/split/ReadVariableOp2b
/lstm_1/while/lstm_cell_1/split_1/ReadVariableOp/lstm_1/while/lstm_cell_1/split_1/ReadVariableOp: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :.*
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ:.*
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ:.*
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ:

_output_shapes
: :

_output_shapes
: :	

_output_shapes
: 
Ý&
ß
@__inference_dense_layer_call_and_return_conditional_losses_17292

inputs%
!tensordot_readvariableop_resource#
biasadd_readvariableop_resource
identity¢BiasAdd/ReadVariableOp¢Tensordot/ReadVariableOp
Tensordot/ReadVariableOpReadVariableOp!tensordot_readvariableop_resource*
_output_shapes
:	7*
dtype02
Tensordot/ReadVariableOpj
Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:2
Tensordot/axesq
Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       2
Tensordot/freeX
Tensordot/ShapeShapeinputs*
T0*
_output_shapes
:2
Tensordot/Shapet
Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : 2
Tensordot/GatherV2/axisÑ
Tensordot/GatherV2GatherV2Tensordot/Shape:output:0Tensordot/free:output:0 Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2
Tensordot/GatherV2x
Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2
Tensordot/GatherV2_1/axis×
Tensordot/GatherV2_1GatherV2Tensordot/Shape:output:0Tensordot/axes:output:0"Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2
Tensordot/GatherV2_1l
Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2
Tensordot/Const
Tensordot/ProdProdTensordot/GatherV2:output:0Tensordot/Const:output:0*
T0*
_output_shapes
: 2
Tensordot/Prodp
Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: 2
Tensordot/Const_1
Tensordot/Prod_1ProdTensordot/GatherV2_1:output:0Tensordot/Const_1:output:0*
T0*
_output_shapes
: 2
Tensordot/Prod_1p
Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 2
Tensordot/concat/axis°
Tensordot/concatConcatV2Tensordot/free:output:0Tensordot/axes:output:0Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:2
Tensordot/concat
Tensordot/stackPackTensordot/Prod:output:0Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:2
Tensordot/stack
Tensordot/transpose	TransposeinputsTensordot/concat:output:0*
T0*5
_output_shapes#
!:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2
Tensordot/transpose
Tensordot/ReshapeReshapeTensordot/transpose:y:0Tensordot/stack:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2
Tensordot/Reshape
Tensordot/MatMulMatMulTensordot/Reshape:output:0 Tensordot/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ72
Tensordot/MatMulp
Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:72
Tensordot/Const_2t
Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2
Tensordot/concat_1/axis½
Tensordot/concat_1ConcatV2Tensordot/GatherV2:output:0Tensordot/Const_2:output:0 Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:2
Tensordot/concat_1
	TensordotReshapeTensordot/MatMul:product:0Tensordot/concat_1:output:0*
T0*4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ72
	Tensordot
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:7*
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddTensordot:output:0BiasAdd/ReadVariableOp:value:0*
T0*4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ72	
BiasAddy
Max/reduction_indicesConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ2
Max/reduction_indices
MaxMaxBiasAdd:output:0Max/reduction_indices:output:0*
T0*4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ*
	keep_dims(2
Maxp
subSubBiasAdd:output:0Max:output:0*
T0*4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ72
subY
ExpExpsub:z:0*
T0*4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ72
Expy
Sum/reduction_indicesConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ2
Sum/reduction_indices
SumSumExp:y:0Sum/reduction_indices:output:0*
T0*4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ*
	keep_dims(2
Sums
truedivRealDivExp:y:0Sum:output:0*
T0*4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ72	
truediv 
IdentityIdentitytruediv:z:0^BiasAdd/ReadVariableOp^Tensordot/ReadVariableOp*
T0*4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ72

Identity"
identityIdentity:output:0*<
_input_shapes+
):ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp24
Tensordot/ReadVariableOpTensordot/ReadVariableOp:] Y
5
_output_shapes#
!:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
÷"

B__inference_model_2_layer_call_and_return_conditional_losses_15153
input_2
input_4
input_3
embedding_1_14436
lstm_1_15088
lstm_1_15090
lstm_1_15092
dense_15145
dense_15147
identity

identity_1

identity_2¢dense/StatefulPartitionedCall¢!dropout_1/StatefulPartitionedCall¢#embedding_1/StatefulPartitionedCall¢lstm_1/StatefulPartitionedCall
#embedding_1/StatefulPartitionedCallStatefulPartitionedCallinput_2embedding_1_14436*
Tin
2*
Tout
2*
_collective_manager_ids
 *5
_output_shapes#
!:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ*#
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *O
fJRH
F__inference_embedding_1_layer_call_and_return_conditional_losses_144272%
#embedding_1/StatefulPartitionedCallu
embedding_1/NotEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *    2
embedding_1/NotEqual/y
embedding_1/NotEqualNotEqualinput_2embedding_1/NotEqual/y:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2
embedding_1/NotEqual
activation_1/PartitionedCallPartitionedCall,embedding_1/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *5
_output_shapes#
!:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *P
fKRI
G__inference_activation_1_layer_call_and_return_conditional_losses_144462
activation_1/PartitionedCall
!dropout_1/StatefulPartitionedCallStatefulPartitionedCall%activation_1/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *5
_output_shapes#
!:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *M
fHRF
D__inference_dropout_1_layer_call_and_return_conditional_losses_144662#
!dropout_1/StatefulPartitionedCall¢
lstm_1/StatefulPartitionedCallStatefulPartitionedCall*dropout_1/StatefulPartitionedCall:output:0embedding_1/NotEqual:z:0input_4input_3lstm_1_15088lstm_1_15090lstm_1_15092*
Tin
	2
*
Tout
2*
_collective_manager_ids
 *]
_output_shapesK
I:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *J
fERC
A__inference_lstm_1_layer_call_and_return_conditional_losses_147992 
lstm_1/StatefulPartitionedCall°
dense/StatefulPartitionedCallStatefulPartitionedCall'lstm_1/StatefulPartitionedCall:output:0dense_15145dense_15147*
Tin
2*
Tout
2*
_collective_manager_ids
 *4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ7*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *I
fDRB
@__inference_dense_layer_call_and_return_conditional_losses_151342
dense/StatefulPartitionedCall
IdentityIdentity&dense/StatefulPartitionedCall:output:0^dense/StatefulPartitionedCall"^dropout_1/StatefulPartitionedCall$^embedding_1/StatefulPartitionedCall^lstm_1/StatefulPartitionedCall*
T0*4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ72

Identity

Identity_1Identity'lstm_1/StatefulPartitionedCall:output:1^dense/StatefulPartitionedCall"^dropout_1/StatefulPartitionedCall$^embedding_1/StatefulPartitionedCall^lstm_1/StatefulPartitionedCall*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity_1

Identity_2Identity'lstm_1/StatefulPartitionedCall:output:2^dense/StatefulPartitionedCall"^dropout_1/StatefulPartitionedCall$^embedding_1/StatefulPartitionedCall^lstm_1/StatefulPartitionedCall*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity_2"
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0*o
_input_shapes^
\:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ::::::2>
dense/StatefulPartitionedCalldense/StatefulPartitionedCall2F
!dropout_1/StatefulPartitionedCall!dropout_1/StatefulPartitionedCall2J
#embedding_1/StatefulPartitionedCall#embedding_1/StatefulPartitionedCall2@
lstm_1/StatefulPartitionedCalllstm_1/StatefulPartitionedCall:Y U
0
_output_shapes
:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
!
_user_specified_name	input_2:QM
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
!
_user_specified_name	input_4:QM
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
!
_user_specified_name	input_3
û
¨

while_body_14902
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_placeholder_4
while_strided_slice_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0[
Wwhile_tensorarrayv2read_1_tensorlistgetitem_tensorarrayunstack_1_tensorlistfromtensor_05
1while_lstm_cell_1_split_readvariableop_resource_07
3while_lstm_cell_1_split_1_readvariableop_resource_0/
+while_lstm_cell_1_readvariableop_resource_0
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_identity_5
while_identity_6
while_strided_sliceU
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorY
Uwhile_tensorarrayv2read_1_tensorlistgetitem_tensorarrayunstack_1_tensorlistfromtensor3
/while_lstm_cell_1_split_readvariableop_resource5
1while_lstm_cell_1_split_1_readvariableop_resource-
)while_lstm_cell_1_readvariableop_resource¢ while/lstm_cell_1/ReadVariableOp¢"while/lstm_cell_1/ReadVariableOp_1¢"while/lstm_cell_1/ReadVariableOp_2¢"while/lstm_cell_1/ReadVariableOp_3¢&while/lstm_cell_1/split/ReadVariableOp¢(while/lstm_cell_1/split_1/ReadVariableOpÃ
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ   29
7while/TensorArrayV2Read/TensorListGetItem/element_shapeÔ
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
element_dtype02+
)while/TensorArrayV2Read/TensorListGetItemÇ
9while/TensorArrayV2Read_1/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ   2;
9while/TensorArrayV2Read_1/TensorListGetItem/element_shapeÝ
+while/TensorArrayV2Read_1/TensorListGetItemTensorListGetItemWwhile_tensorarrayv2read_1_tensorlistgetitem_tensorarrayunstack_1_tensorlistfromtensor_0while_placeholderBwhile/TensorArrayV2Read_1/TensorListGetItem/element_shape:output:0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
element_dtype0
2-
+while/TensorArrayV2Read_1/TensorListGetItem
!while/lstm_cell_1/ones_like/ShapeShapewhile_placeholder_3*
T0*
_output_shapes
:2#
!while/lstm_cell_1/ones_like/Shape
!while/lstm_cell_1/ones_like/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ?2#
!while/lstm_cell_1/ones_like/ConstÍ
while/lstm_cell_1/ones_likeFill*while/lstm_cell_1/ones_like/Shape:output:0*while/lstm_cell_1/ones_like/Const:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
while/lstm_cell_1/ones_liket
while/lstm_cell_1/ConstConst*
_output_shapes
: *
dtype0*
value	B :2
while/lstm_cell_1/Const
!while/lstm_cell_1/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2#
!while/lstm_cell_1/split/split_dimÄ
&while/lstm_cell_1/split/ReadVariableOpReadVariableOp1while_lstm_cell_1_split_readvariableop_resource_0* 
_output_shapes
:
*
dtype02(
&while/lstm_cell_1/split/ReadVariableOp÷
while/lstm_cell_1/splitSplit*while/lstm_cell_1/split/split_dim:output:0.while/lstm_cell_1/split/ReadVariableOp:value:0*
T0*D
_output_shapes2
0:
:
:
:
*
	num_split2
while/lstm_cell_1/splitÅ
while/lstm_cell_1/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:0 while/lstm_cell_1/split:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
while/lstm_cell_1/MatMulÉ
while/lstm_cell_1/MatMul_1MatMul0while/TensorArrayV2Read/TensorListGetItem:item:0 while/lstm_cell_1/split:output:1*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
while/lstm_cell_1/MatMul_1É
while/lstm_cell_1/MatMul_2MatMul0while/TensorArrayV2Read/TensorListGetItem:item:0 while/lstm_cell_1/split:output:2*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
while/lstm_cell_1/MatMul_2É
while/lstm_cell_1/MatMul_3MatMul0while/TensorArrayV2Read/TensorListGetItem:item:0 while/lstm_cell_1/split:output:3*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
while/lstm_cell_1/MatMul_3x
while/lstm_cell_1/Const_1Const*
_output_shapes
: *
dtype0*
value	B :2
while/lstm_cell_1/Const_1
#while/lstm_cell_1/split_1/split_dimConst*
_output_shapes
: *
dtype0*
value	B : 2%
#while/lstm_cell_1/split_1/split_dimÅ
(while/lstm_cell_1/split_1/ReadVariableOpReadVariableOp3while_lstm_cell_1_split_1_readvariableop_resource_0*
_output_shapes	
:*
dtype02*
(while/lstm_cell_1/split_1/ReadVariableOpë
while/lstm_cell_1/split_1Split,while/lstm_cell_1/split_1/split_dim:output:00while/lstm_cell_1/split_1/ReadVariableOp:value:0*
T0*0
_output_shapes
::::*
	num_split2
while/lstm_cell_1/split_1¼
while/lstm_cell_1/BiasAddBiasAdd"while/lstm_cell_1/MatMul:product:0"while/lstm_cell_1/split_1:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
while/lstm_cell_1/BiasAddÂ
while/lstm_cell_1/BiasAdd_1BiasAdd$while/lstm_cell_1/MatMul_1:product:0"while/lstm_cell_1/split_1:output:1*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
while/lstm_cell_1/BiasAdd_1Â
while/lstm_cell_1/BiasAdd_2BiasAdd$while/lstm_cell_1/MatMul_2:product:0"while/lstm_cell_1/split_1:output:2*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
while/lstm_cell_1/BiasAdd_2Â
while/lstm_cell_1/BiasAdd_3BiasAdd$while/lstm_cell_1/MatMul_3:product:0"while/lstm_cell_1/split_1:output:3*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
while/lstm_cell_1/BiasAdd_3£
while/lstm_cell_1/mulMulwhile_placeholder_3$while/lstm_cell_1/ones_like:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
while/lstm_cell_1/mul§
while/lstm_cell_1/mul_1Mulwhile_placeholder_3$while/lstm_cell_1/ones_like:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
while/lstm_cell_1/mul_1§
while/lstm_cell_1/mul_2Mulwhile_placeholder_3$while/lstm_cell_1/ones_like:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
while/lstm_cell_1/mul_2§
while/lstm_cell_1/mul_3Mulwhile_placeholder_3$while/lstm_cell_1/ones_like:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
while/lstm_cell_1/mul_3²
 while/lstm_cell_1/ReadVariableOpReadVariableOp+while_lstm_cell_1_readvariableop_resource_0* 
_output_shapes
:
*
dtype02"
 while/lstm_cell_1/ReadVariableOp
%while/lstm_cell_1/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB"        2'
%while/lstm_cell_1/strided_slice/stack£
'while/lstm_cell_1/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       2)
'while/lstm_cell_1/strided_slice/stack_1£
'while/lstm_cell_1/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2)
'while/lstm_cell_1/strided_slice/stack_2ê
while/lstm_cell_1/strided_sliceStridedSlice(while/lstm_cell_1/ReadVariableOp:value:0.while/lstm_cell_1/strided_slice/stack:output:00while/lstm_cell_1/strided_slice/stack_1:output:00while/lstm_cell_1/strided_slice/stack_2:output:0*
Index0*
T0* 
_output_shapes
:
*

begin_mask*
end_mask2!
while/lstm_cell_1/strided_sliceº
while/lstm_cell_1/MatMul_4MatMulwhile/lstm_cell_1/mul:z:0(while/lstm_cell_1/strided_slice:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
while/lstm_cell_1/MatMul_4´
while/lstm_cell_1/addAddV2"while/lstm_cell_1/BiasAdd:output:0$while/lstm_cell_1/MatMul_4:product:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
while/lstm_cell_1/add
while/lstm_cell_1/SigmoidSigmoidwhile/lstm_cell_1/add:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
while/lstm_cell_1/Sigmoid¶
"while/lstm_cell_1/ReadVariableOp_1ReadVariableOp+while_lstm_cell_1_readvariableop_resource_0* 
_output_shapes
:
*
dtype02$
"while/lstm_cell_1/ReadVariableOp_1£
'while/lstm_cell_1/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB"       2)
'while/lstm_cell_1/strided_slice_1/stack§
)while/lstm_cell_1/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       2+
)while/lstm_cell_1/strided_slice_1/stack_1§
)while/lstm_cell_1/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2+
)while/lstm_cell_1/strided_slice_1/stack_2ö
!while/lstm_cell_1/strided_slice_1StridedSlice*while/lstm_cell_1/ReadVariableOp_1:value:00while/lstm_cell_1/strided_slice_1/stack:output:02while/lstm_cell_1/strided_slice_1/stack_1:output:02while/lstm_cell_1/strided_slice_1/stack_2:output:0*
Index0*
T0* 
_output_shapes
:
*

begin_mask*
end_mask2#
!while/lstm_cell_1/strided_slice_1¾
while/lstm_cell_1/MatMul_5MatMulwhile/lstm_cell_1/mul_1:z:0*while/lstm_cell_1/strided_slice_1:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
while/lstm_cell_1/MatMul_5º
while/lstm_cell_1/add_1AddV2$while/lstm_cell_1/BiasAdd_1:output:0$while/lstm_cell_1/MatMul_5:product:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
while/lstm_cell_1/add_1
while/lstm_cell_1/Sigmoid_1Sigmoidwhile/lstm_cell_1/add_1:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
while/lstm_cell_1/Sigmoid_1¢
while/lstm_cell_1/mul_4Mulwhile/lstm_cell_1/Sigmoid_1:y:0while_placeholder_4*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
while/lstm_cell_1/mul_4¶
"while/lstm_cell_1/ReadVariableOp_2ReadVariableOp+while_lstm_cell_1_readvariableop_resource_0* 
_output_shapes
:
*
dtype02$
"while/lstm_cell_1/ReadVariableOp_2£
'while/lstm_cell_1/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB"       2)
'while/lstm_cell_1/strided_slice_2/stack§
)while/lstm_cell_1/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       2+
)while/lstm_cell_1/strided_slice_2/stack_1§
)while/lstm_cell_1/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2+
)while/lstm_cell_1/strided_slice_2/stack_2ö
!while/lstm_cell_1/strided_slice_2StridedSlice*while/lstm_cell_1/ReadVariableOp_2:value:00while/lstm_cell_1/strided_slice_2/stack:output:02while/lstm_cell_1/strided_slice_2/stack_1:output:02while/lstm_cell_1/strided_slice_2/stack_2:output:0*
Index0*
T0* 
_output_shapes
:
*

begin_mask*
end_mask2#
!while/lstm_cell_1/strided_slice_2¾
while/lstm_cell_1/MatMul_6MatMulwhile/lstm_cell_1/mul_2:z:0*while/lstm_cell_1/strided_slice_2:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
while/lstm_cell_1/MatMul_6º
while/lstm_cell_1/add_2AddV2$while/lstm_cell_1/BiasAdd_2:output:0$while/lstm_cell_1/MatMul_6:product:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
while/lstm_cell_1/add_2
while/lstm_cell_1/TanhTanhwhile/lstm_cell_1/add_2:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
while/lstm_cell_1/Tanh§
while/lstm_cell_1/mul_5Mulwhile/lstm_cell_1/Sigmoid:y:0while/lstm_cell_1/Tanh:y:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
while/lstm_cell_1/mul_5¨
while/lstm_cell_1/add_3AddV2while/lstm_cell_1/mul_4:z:0while/lstm_cell_1/mul_5:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
while/lstm_cell_1/add_3¶
"while/lstm_cell_1/ReadVariableOp_3ReadVariableOp+while_lstm_cell_1_readvariableop_resource_0* 
_output_shapes
:
*
dtype02$
"while/lstm_cell_1/ReadVariableOp_3£
'while/lstm_cell_1/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB"       2)
'while/lstm_cell_1/strided_slice_3/stack§
)while/lstm_cell_1/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        2+
)while/lstm_cell_1/strided_slice_3/stack_1§
)while/lstm_cell_1/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2+
)while/lstm_cell_1/strided_slice_3/stack_2ö
!while/lstm_cell_1/strided_slice_3StridedSlice*while/lstm_cell_1/ReadVariableOp_3:value:00while/lstm_cell_1/strided_slice_3/stack:output:02while/lstm_cell_1/strided_slice_3/stack_1:output:02while/lstm_cell_1/strided_slice_3/stack_2:output:0*
Index0*
T0* 
_output_shapes
:
*

begin_mask*
end_mask2#
!while/lstm_cell_1/strided_slice_3¾
while/lstm_cell_1/MatMul_7MatMulwhile/lstm_cell_1/mul_3:z:0*while/lstm_cell_1/strided_slice_3:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
while/lstm_cell_1/MatMul_7º
while/lstm_cell_1/add_4AddV2$while/lstm_cell_1/BiasAdd_3:output:0$while/lstm_cell_1/MatMul_7:product:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
while/lstm_cell_1/add_4
while/lstm_cell_1/Sigmoid_2Sigmoidwhile/lstm_cell_1/add_4:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
while/lstm_cell_1/Sigmoid_2
while/lstm_cell_1/Tanh_1Tanhwhile/lstm_cell_1/add_3:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
while/lstm_cell_1/Tanh_1«
while/lstm_cell_1/mul_6Mulwhile/lstm_cell_1/Sigmoid_2:y:0while/lstm_cell_1/Tanh_1:y:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
while/lstm_cell_1/mul_6}
while/Tile/multiplesConst*
_output_shapes
:*
dtype0*
valueB"      2
while/Tile/multiples¥

while/TileTile2while/TensorArrayV2Read_1/TensorListGetItem:item:0while/Tile/multiples:output:0*
T0
*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

while/Tile¦
while/SelectV2SelectV2while/Tile:output:0while/lstm_cell_1/mul_6:z:0while_placeholder_2*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
while/SelectV2
while/Tile_1/multiplesConst*
_output_shapes
:*
dtype0*
valueB"      2
while/Tile_1/multiples«
while/Tile_1Tile2while/TensorArrayV2Read_1/TensorListGetItem:item:0while/Tile_1/multiples:output:0*
T0
*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
while/Tile_1
while/Tile_2/multiplesConst*
_output_shapes
:*
dtype0*
valueB"      2
while/Tile_2/multiples«
while/Tile_2Tile2while/TensorArrayV2Read_1/TensorListGetItem:item:0while/Tile_2/multiples:output:0*
T0
*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
while/Tile_2¬
while/SelectV2_1SelectV2while/Tile_1:output:0while/lstm_cell_1/mul_6:z:0while_placeholder_3*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
while/SelectV2_1¬
while/SelectV2_2SelectV2while/Tile_2:output:0while/lstm_cell_1/add_3:z:0while_placeholder_4*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
while/SelectV2_2Û
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholderwhile/SelectV2:output:0*
_output_shapes
: *
element_dtype02,
*while/TensorArrayV2Write/TensorListSetItem\
while/add/yConst*
_output_shapes
: *
dtype0*
value	B :2
while/add/yi
	while/addAddV2while_placeholderwhile/add/y:output:0*
T0*
_output_shapes
: 2
	while/add`
while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :2
while/add_1/yv
while/add_1AddV2while_while_loop_counterwhile/add_1/y:output:0*
T0*
_output_shapes
: 2
while/add_1Ä
while/IdentityIdentitywhile/add_1:z:0!^while/lstm_cell_1/ReadVariableOp#^while/lstm_cell_1/ReadVariableOp_1#^while/lstm_cell_1/ReadVariableOp_2#^while/lstm_cell_1/ReadVariableOp_3'^while/lstm_cell_1/split/ReadVariableOp)^while/lstm_cell_1/split_1/ReadVariableOp*
T0*
_output_shapes
: 2
while/Identity×
while/Identity_1Identitywhile_while_maximum_iterations!^while/lstm_cell_1/ReadVariableOp#^while/lstm_cell_1/ReadVariableOp_1#^while/lstm_cell_1/ReadVariableOp_2#^while/lstm_cell_1/ReadVariableOp_3'^while/lstm_cell_1/split/ReadVariableOp)^while/lstm_cell_1/split_1/ReadVariableOp*
T0*
_output_shapes
: 2
while/Identity_1Æ
while/Identity_2Identitywhile/add:z:0!^while/lstm_cell_1/ReadVariableOp#^while/lstm_cell_1/ReadVariableOp_1#^while/lstm_cell_1/ReadVariableOp_2#^while/lstm_cell_1/ReadVariableOp_3'^while/lstm_cell_1/split/ReadVariableOp)^while/lstm_cell_1/split_1/ReadVariableOp*
T0*
_output_shapes
: 2
while/Identity_2ó
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0!^while/lstm_cell_1/ReadVariableOp#^while/lstm_cell_1/ReadVariableOp_1#^while/lstm_cell_1/ReadVariableOp_2#^while/lstm_cell_1/ReadVariableOp_3'^while/lstm_cell_1/split/ReadVariableOp)^while/lstm_cell_1/split_1/ReadVariableOp*
T0*
_output_shapes
: 2
while/Identity_3â
while/Identity_4Identitywhile/SelectV2:output:0!^while/lstm_cell_1/ReadVariableOp#^while/lstm_cell_1/ReadVariableOp_1#^while/lstm_cell_1/ReadVariableOp_2#^while/lstm_cell_1/ReadVariableOp_3'^while/lstm_cell_1/split/ReadVariableOp)^while/lstm_cell_1/split_1/ReadVariableOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
while/Identity_4ä
while/Identity_5Identitywhile/SelectV2_1:output:0!^while/lstm_cell_1/ReadVariableOp#^while/lstm_cell_1/ReadVariableOp_1#^while/lstm_cell_1/ReadVariableOp_2#^while/lstm_cell_1/ReadVariableOp_3'^while/lstm_cell_1/split/ReadVariableOp)^while/lstm_cell_1/split_1/ReadVariableOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
while/Identity_5ä
while/Identity_6Identitywhile/SelectV2_2:output:0!^while/lstm_cell_1/ReadVariableOp#^while/lstm_cell_1/ReadVariableOp_1#^while/lstm_cell_1/ReadVariableOp_2#^while/lstm_cell_1/ReadVariableOp_3'^while/lstm_cell_1/split/ReadVariableOp)^while/lstm_cell_1/split_1/ReadVariableOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
while/Identity_6")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"-
while_identity_5while/Identity_5:output:0"-
while_identity_6while/Identity_6:output:0"X
)while_lstm_cell_1_readvariableop_resource+while_lstm_cell_1_readvariableop_resource_0"h
1while_lstm_cell_1_split_1_readvariableop_resource3while_lstm_cell_1_split_1_readvariableop_resource_0"d
/while_lstm_cell_1_split_readvariableop_resource1while_lstm_cell_1_split_readvariableop_resource_0",
while_strided_slicewhile_strided_slice_0"°
Uwhile_tensorarrayv2read_1_tensorlistgetitem_tensorarrayunstack_1_tensorlistfromtensorWwhile_tensorarrayv2read_1_tensorlistgetitem_tensorarrayunstack_1_tensorlistfromtensor_0"¨
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*i
_input_shapesX
V: : : : :ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ: : : :::2D
 while/lstm_cell_1/ReadVariableOp while/lstm_cell_1/ReadVariableOp2H
"while/lstm_cell_1/ReadVariableOp_1"while/lstm_cell_1/ReadVariableOp_12H
"while/lstm_cell_1/ReadVariableOp_2"while/lstm_cell_1/ReadVariableOp_22H
"while/lstm_cell_1/ReadVariableOp_3"while/lstm_cell_1/ReadVariableOp_32P
&while/lstm_cell_1/split/ReadVariableOp&while/lstm_cell_1/split/ReadVariableOp2T
(while/lstm_cell_1/split_1/ReadVariableOp(while/lstm_cell_1/split_1/ReadVariableOp: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :.*
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ:.*
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ:.*
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ:

_output_shapes
: :

_output_shapes
: :	

_output_shapes
: 
®
ò
 __inference__wrapped_model_13665
input_2
input_4
input_3.
*model_2_embedding_1_embedding_lookup_13377<
8model_2_lstm_1_lstm_cell_1_split_readvariableop_resource>
:model_2_lstm_1_lstm_cell_1_split_1_readvariableop_resource6
2model_2_lstm_1_lstm_cell_1_readvariableop_resource3
/model_2_dense_tensordot_readvariableop_resource1
-model_2_dense_biasadd_readvariableop_resource
identity

identity_1

identity_2¢$model_2/dense/BiasAdd/ReadVariableOp¢&model_2/dense/Tensordot/ReadVariableOp¢$model_2/embedding_1/embedding_lookup¢)model_2/lstm_1/lstm_cell_1/ReadVariableOp¢+model_2/lstm_1/lstm_cell_1/ReadVariableOp_1¢+model_2/lstm_1/lstm_cell_1/ReadVariableOp_2¢+model_2/lstm_1/lstm_cell_1/ReadVariableOp_3¢/model_2/lstm_1/lstm_cell_1/split/ReadVariableOp¢1model_2/lstm_1/lstm_cell_1/split_1/ReadVariableOp¢model_2/lstm_1/while
model_2/embedding_1/CastCastinput_2*

DstT0*

SrcT0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2
model_2/embedding_1/Casté
$model_2/embedding_1/embedding_lookupResourceGather*model_2_embedding_1_embedding_lookup_13377model_2/embedding_1/Cast:y:0",/job:localhost/replica:0/task:0/device:CPU:0*
Tindices0*=
_class3
1/loc:@model_2/embedding_1/embedding_lookup/13377*5
_output_shapes#
!:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ*
dtype02&
$model_2/embedding_1/embedding_lookupÆ
-model_2/embedding_1/embedding_lookup/IdentityIdentity-model_2/embedding_1/embedding_lookup:output:0",/job:localhost/replica:0/task:0/device:CPU:0*
T0*=
_class3
1/loc:@model_2/embedding_1/embedding_lookup/13377*5
_output_shapes#
!:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2/
-model_2/embedding_1/embedding_lookup/Identityæ
/model_2/embedding_1/embedding_lookup/Identity_1Identity6model_2/embedding_1/embedding_lookup/Identity:output:0*
T0*5
_output_shapes#
!:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ21
/model_2/embedding_1/embedding_lookup/Identity_1
model_2/embedding_1/NotEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *    2 
model_2/embedding_1/NotEqual/yµ
model_2/embedding_1/NotEqualNotEqualinput_2'model_2/embedding_1/NotEqual/y:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2
model_2/embedding_1/NotEqual¸
model_2/activation_1/ReluRelu8model_2/embedding_1/embedding_lookup/Identity_1:output:0*
T0*5
_output_shapes#
!:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2
model_2/activation_1/Relu­
model_2/dropout_1/IdentityIdentity'model_2/activation_1/Relu:activations:0*
T0*5
_output_shapes#
!:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2
model_2/dropout_1/Identity
model_2/lstm_1/transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
model_2/lstm_1/transpose/permÎ
model_2/lstm_1/transpose	Transpose#model_2/dropout_1/Identity:output:0&model_2/lstm_1/transpose/perm:output:0*
T0*5
_output_shapes#
!:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2
model_2/lstm_1/transposex
model_2/lstm_1/ShapeShapemodel_2/lstm_1/transpose:y:0*
T0*
_output_shapes
:2
model_2/lstm_1/Shape
"model_2/lstm_1/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2$
"model_2/lstm_1/strided_slice/stack
$model_2/lstm_1/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2&
$model_2/lstm_1/strided_slice/stack_1
$model_2/lstm_1/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2&
$model_2/lstm_1/strided_slice/stack_2¼
model_2/lstm_1/strided_sliceStridedSlicemodel_2/lstm_1/Shape:output:0+model_2/lstm_1/strided_slice/stack:output:0-model_2/lstm_1/strided_slice/stack_1:output:0-model_2/lstm_1/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
model_2/lstm_1/strided_slice
model_2/lstm_1/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ2
model_2/lstm_1/ExpandDims/dimÍ
model_2/lstm_1/ExpandDims
ExpandDims model_2/embedding_1/NotEqual:z:0&model_2/lstm_1/ExpandDims/dim:output:0*
T0
*4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2
model_2/lstm_1/ExpandDims
model_2/lstm_1/transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          2!
model_2/lstm_1/transpose_1/permÒ
model_2/lstm_1/transpose_1	Transpose"model_2/lstm_1/ExpandDims:output:0(model_2/lstm_1/transpose_1/perm:output:0*
T0
*4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2
model_2/lstm_1/transpose_1£
*model_2/lstm_1/TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ2,
*model_2/lstm_1/TensorArrayV2/element_shapeì
model_2/lstm_1/TensorArrayV2TensorListReserve3model_2/lstm_1/TensorArrayV2/element_shape:output:0%model_2/lstm_1/strided_slice:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
model_2/lstm_1/TensorArrayV2Ý
Dmodel_2/lstm_1/TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ   2F
Dmodel_2/lstm_1/TensorArrayUnstack/TensorListFromTensor/element_shape´
6model_2/lstm_1/TensorArrayUnstack/TensorListFromTensorTensorListFromTensormodel_2/lstm_1/transpose:y:0Mmodel_2/lstm_1/TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type028
6model_2/lstm_1/TensorArrayUnstack/TensorListFromTensor
$model_2/lstm_1/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2&
$model_2/lstm_1/strided_slice_1/stack
&model_2/lstm_1/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2(
&model_2/lstm_1/strided_slice_1/stack_1
&model_2/lstm_1/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2(
&model_2/lstm_1/strided_slice_1/stack_2×
model_2/lstm_1/strided_slice_1StridedSlicemodel_2/lstm_1/transpose:y:0-model_2/lstm_1/strided_slice_1/stack:output:0/model_2/lstm_1/strided_slice_1/stack_1:output:0/model_2/lstm_1/strided_slice_1/stack_2:output:0*
Index0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
shrink_axis_mask2 
model_2/lstm_1/strided_slice_1
*model_2/lstm_1/lstm_cell_1/ones_like/ShapeShapeinput_4*
T0*
_output_shapes
:2,
*model_2/lstm_1/lstm_cell_1/ones_like/Shape
*model_2/lstm_1/lstm_cell_1/ones_like/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ?2,
*model_2/lstm_1/lstm_cell_1/ones_like/Constñ
$model_2/lstm_1/lstm_cell_1/ones_likeFill3model_2/lstm_1/lstm_cell_1/ones_like/Shape:output:03model_2/lstm_1/lstm_cell_1/ones_like/Const:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2&
$model_2/lstm_1/lstm_cell_1/ones_like
 model_2/lstm_1/lstm_cell_1/ConstConst*
_output_shapes
: *
dtype0*
value	B :2"
 model_2/lstm_1/lstm_cell_1/Const
*model_2/lstm_1/lstm_cell_1/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2,
*model_2/lstm_1/lstm_cell_1/split/split_dimÝ
/model_2/lstm_1/lstm_cell_1/split/ReadVariableOpReadVariableOp8model_2_lstm_1_lstm_cell_1_split_readvariableop_resource* 
_output_shapes
:
*
dtype021
/model_2/lstm_1/lstm_cell_1/split/ReadVariableOp
 model_2/lstm_1/lstm_cell_1/splitSplit3model_2/lstm_1/lstm_cell_1/split/split_dim:output:07model_2/lstm_1/lstm_cell_1/split/ReadVariableOp:value:0*
T0*D
_output_shapes2
0:
:
:
:
*
	num_split2"
 model_2/lstm_1/lstm_cell_1/split×
!model_2/lstm_1/lstm_cell_1/MatMulMatMul'model_2/lstm_1/strided_slice_1:output:0)model_2/lstm_1/lstm_cell_1/split:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2#
!model_2/lstm_1/lstm_cell_1/MatMulÛ
#model_2/lstm_1/lstm_cell_1/MatMul_1MatMul'model_2/lstm_1/strided_slice_1:output:0)model_2/lstm_1/lstm_cell_1/split:output:1*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2%
#model_2/lstm_1/lstm_cell_1/MatMul_1Û
#model_2/lstm_1/lstm_cell_1/MatMul_2MatMul'model_2/lstm_1/strided_slice_1:output:0)model_2/lstm_1/lstm_cell_1/split:output:2*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2%
#model_2/lstm_1/lstm_cell_1/MatMul_2Û
#model_2/lstm_1/lstm_cell_1/MatMul_3MatMul'model_2/lstm_1/strided_slice_1:output:0)model_2/lstm_1/lstm_cell_1/split:output:3*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2%
#model_2/lstm_1/lstm_cell_1/MatMul_3
"model_2/lstm_1/lstm_cell_1/Const_1Const*
_output_shapes
: *
dtype0*
value	B :2$
"model_2/lstm_1/lstm_cell_1/Const_1
,model_2/lstm_1/lstm_cell_1/split_1/split_dimConst*
_output_shapes
: *
dtype0*
value	B : 2.
,model_2/lstm_1/lstm_cell_1/split_1/split_dimÞ
1model_2/lstm_1/lstm_cell_1/split_1/ReadVariableOpReadVariableOp:model_2_lstm_1_lstm_cell_1_split_1_readvariableop_resource*
_output_shapes	
:*
dtype023
1model_2/lstm_1/lstm_cell_1/split_1/ReadVariableOp
"model_2/lstm_1/lstm_cell_1/split_1Split5model_2/lstm_1/lstm_cell_1/split_1/split_dim:output:09model_2/lstm_1/lstm_cell_1/split_1/ReadVariableOp:value:0*
T0*0
_output_shapes
::::*
	num_split2$
"model_2/lstm_1/lstm_cell_1/split_1à
"model_2/lstm_1/lstm_cell_1/BiasAddBiasAdd+model_2/lstm_1/lstm_cell_1/MatMul:product:0+model_2/lstm_1/lstm_cell_1/split_1:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2$
"model_2/lstm_1/lstm_cell_1/BiasAddæ
$model_2/lstm_1/lstm_cell_1/BiasAdd_1BiasAdd-model_2/lstm_1/lstm_cell_1/MatMul_1:product:0+model_2/lstm_1/lstm_cell_1/split_1:output:1*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2&
$model_2/lstm_1/lstm_cell_1/BiasAdd_1æ
$model_2/lstm_1/lstm_cell_1/BiasAdd_2BiasAdd-model_2/lstm_1/lstm_cell_1/MatMul_2:product:0+model_2/lstm_1/lstm_cell_1/split_1:output:2*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2&
$model_2/lstm_1/lstm_cell_1/BiasAdd_2æ
$model_2/lstm_1/lstm_cell_1/BiasAdd_3BiasAdd-model_2/lstm_1/lstm_cell_1/MatMul_3:product:0+model_2/lstm_1/lstm_cell_1/split_1:output:3*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2&
$model_2/lstm_1/lstm_cell_1/BiasAdd_3²
model_2/lstm_1/lstm_cell_1/mulMulinput_4-model_2/lstm_1/lstm_cell_1/ones_like:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2 
model_2/lstm_1/lstm_cell_1/mul¶
 model_2/lstm_1/lstm_cell_1/mul_1Mulinput_4-model_2/lstm_1/lstm_cell_1/ones_like:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2"
 model_2/lstm_1/lstm_cell_1/mul_1¶
 model_2/lstm_1/lstm_cell_1/mul_2Mulinput_4-model_2/lstm_1/lstm_cell_1/ones_like:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2"
 model_2/lstm_1/lstm_cell_1/mul_2¶
 model_2/lstm_1/lstm_cell_1/mul_3Mulinput_4-model_2/lstm_1/lstm_cell_1/ones_like:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2"
 model_2/lstm_1/lstm_cell_1/mul_3Ë
)model_2/lstm_1/lstm_cell_1/ReadVariableOpReadVariableOp2model_2_lstm_1_lstm_cell_1_readvariableop_resource* 
_output_shapes
:
*
dtype02+
)model_2/lstm_1/lstm_cell_1/ReadVariableOp±
.model_2/lstm_1/lstm_cell_1/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB"        20
.model_2/lstm_1/lstm_cell_1/strided_slice/stackµ
0model_2/lstm_1/lstm_cell_1/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       22
0model_2/lstm_1/lstm_cell_1/strided_slice/stack_1µ
0model_2/lstm_1/lstm_cell_1/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      22
0model_2/lstm_1/lstm_cell_1/strided_slice/stack_2 
(model_2/lstm_1/lstm_cell_1/strided_sliceStridedSlice1model_2/lstm_1/lstm_cell_1/ReadVariableOp:value:07model_2/lstm_1/lstm_cell_1/strided_slice/stack:output:09model_2/lstm_1/lstm_cell_1/strided_slice/stack_1:output:09model_2/lstm_1/lstm_cell_1/strided_slice/stack_2:output:0*
Index0*
T0* 
_output_shapes
:
*

begin_mask*
end_mask2*
(model_2/lstm_1/lstm_cell_1/strided_sliceÞ
#model_2/lstm_1/lstm_cell_1/MatMul_4MatMul"model_2/lstm_1/lstm_cell_1/mul:z:01model_2/lstm_1/lstm_cell_1/strided_slice:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2%
#model_2/lstm_1/lstm_cell_1/MatMul_4Ø
model_2/lstm_1/lstm_cell_1/addAddV2+model_2/lstm_1/lstm_cell_1/BiasAdd:output:0-model_2/lstm_1/lstm_cell_1/MatMul_4:product:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2 
model_2/lstm_1/lstm_cell_1/addª
"model_2/lstm_1/lstm_cell_1/SigmoidSigmoid"model_2/lstm_1/lstm_cell_1/add:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2$
"model_2/lstm_1/lstm_cell_1/SigmoidÏ
+model_2/lstm_1/lstm_cell_1/ReadVariableOp_1ReadVariableOp2model_2_lstm_1_lstm_cell_1_readvariableop_resource* 
_output_shapes
:
*
dtype02-
+model_2/lstm_1/lstm_cell_1/ReadVariableOp_1µ
0model_2/lstm_1/lstm_cell_1/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB"       22
0model_2/lstm_1/lstm_cell_1/strided_slice_1/stack¹
2model_2/lstm_1/lstm_cell_1/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       24
2model_2/lstm_1/lstm_cell_1/strided_slice_1/stack_1¹
2model_2/lstm_1/lstm_cell_1/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      24
2model_2/lstm_1/lstm_cell_1/strided_slice_1/stack_2¬
*model_2/lstm_1/lstm_cell_1/strided_slice_1StridedSlice3model_2/lstm_1/lstm_cell_1/ReadVariableOp_1:value:09model_2/lstm_1/lstm_cell_1/strided_slice_1/stack:output:0;model_2/lstm_1/lstm_cell_1/strided_slice_1/stack_1:output:0;model_2/lstm_1/lstm_cell_1/strided_slice_1/stack_2:output:0*
Index0*
T0* 
_output_shapes
:
*

begin_mask*
end_mask2,
*model_2/lstm_1/lstm_cell_1/strided_slice_1â
#model_2/lstm_1/lstm_cell_1/MatMul_5MatMul$model_2/lstm_1/lstm_cell_1/mul_1:z:03model_2/lstm_1/lstm_cell_1/strided_slice_1:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2%
#model_2/lstm_1/lstm_cell_1/MatMul_5Þ
 model_2/lstm_1/lstm_cell_1/add_1AddV2-model_2/lstm_1/lstm_cell_1/BiasAdd_1:output:0-model_2/lstm_1/lstm_cell_1/MatMul_5:product:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2"
 model_2/lstm_1/lstm_cell_1/add_1°
$model_2/lstm_1/lstm_cell_1/Sigmoid_1Sigmoid$model_2/lstm_1/lstm_cell_1/add_1:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2&
$model_2/lstm_1/lstm_cell_1/Sigmoid_1±
 model_2/lstm_1/lstm_cell_1/mul_4Mul(model_2/lstm_1/lstm_cell_1/Sigmoid_1:y:0input_3*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2"
 model_2/lstm_1/lstm_cell_1/mul_4Ï
+model_2/lstm_1/lstm_cell_1/ReadVariableOp_2ReadVariableOp2model_2_lstm_1_lstm_cell_1_readvariableop_resource* 
_output_shapes
:
*
dtype02-
+model_2/lstm_1/lstm_cell_1/ReadVariableOp_2µ
0model_2/lstm_1/lstm_cell_1/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB"       22
0model_2/lstm_1/lstm_cell_1/strided_slice_2/stack¹
2model_2/lstm_1/lstm_cell_1/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       24
2model_2/lstm_1/lstm_cell_1/strided_slice_2/stack_1¹
2model_2/lstm_1/lstm_cell_1/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      24
2model_2/lstm_1/lstm_cell_1/strided_slice_2/stack_2¬
*model_2/lstm_1/lstm_cell_1/strided_slice_2StridedSlice3model_2/lstm_1/lstm_cell_1/ReadVariableOp_2:value:09model_2/lstm_1/lstm_cell_1/strided_slice_2/stack:output:0;model_2/lstm_1/lstm_cell_1/strided_slice_2/stack_1:output:0;model_2/lstm_1/lstm_cell_1/strided_slice_2/stack_2:output:0*
Index0*
T0* 
_output_shapes
:
*

begin_mask*
end_mask2,
*model_2/lstm_1/lstm_cell_1/strided_slice_2â
#model_2/lstm_1/lstm_cell_1/MatMul_6MatMul$model_2/lstm_1/lstm_cell_1/mul_2:z:03model_2/lstm_1/lstm_cell_1/strided_slice_2:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2%
#model_2/lstm_1/lstm_cell_1/MatMul_6Þ
 model_2/lstm_1/lstm_cell_1/add_2AddV2-model_2/lstm_1/lstm_cell_1/BiasAdd_2:output:0-model_2/lstm_1/lstm_cell_1/MatMul_6:product:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2"
 model_2/lstm_1/lstm_cell_1/add_2£
model_2/lstm_1/lstm_cell_1/TanhTanh$model_2/lstm_1/lstm_cell_1/add_2:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2!
model_2/lstm_1/lstm_cell_1/TanhË
 model_2/lstm_1/lstm_cell_1/mul_5Mul&model_2/lstm_1/lstm_cell_1/Sigmoid:y:0#model_2/lstm_1/lstm_cell_1/Tanh:y:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2"
 model_2/lstm_1/lstm_cell_1/mul_5Ì
 model_2/lstm_1/lstm_cell_1/add_3AddV2$model_2/lstm_1/lstm_cell_1/mul_4:z:0$model_2/lstm_1/lstm_cell_1/mul_5:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2"
 model_2/lstm_1/lstm_cell_1/add_3Ï
+model_2/lstm_1/lstm_cell_1/ReadVariableOp_3ReadVariableOp2model_2_lstm_1_lstm_cell_1_readvariableop_resource* 
_output_shapes
:
*
dtype02-
+model_2/lstm_1/lstm_cell_1/ReadVariableOp_3µ
0model_2/lstm_1/lstm_cell_1/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB"       22
0model_2/lstm_1/lstm_cell_1/strided_slice_3/stack¹
2model_2/lstm_1/lstm_cell_1/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        24
2model_2/lstm_1/lstm_cell_1/strided_slice_3/stack_1¹
2model_2/lstm_1/lstm_cell_1/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      24
2model_2/lstm_1/lstm_cell_1/strided_slice_3/stack_2¬
*model_2/lstm_1/lstm_cell_1/strided_slice_3StridedSlice3model_2/lstm_1/lstm_cell_1/ReadVariableOp_3:value:09model_2/lstm_1/lstm_cell_1/strided_slice_3/stack:output:0;model_2/lstm_1/lstm_cell_1/strided_slice_3/stack_1:output:0;model_2/lstm_1/lstm_cell_1/strided_slice_3/stack_2:output:0*
Index0*
T0* 
_output_shapes
:
*

begin_mask*
end_mask2,
*model_2/lstm_1/lstm_cell_1/strided_slice_3â
#model_2/lstm_1/lstm_cell_1/MatMul_7MatMul$model_2/lstm_1/lstm_cell_1/mul_3:z:03model_2/lstm_1/lstm_cell_1/strided_slice_3:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2%
#model_2/lstm_1/lstm_cell_1/MatMul_7Þ
 model_2/lstm_1/lstm_cell_1/add_4AddV2-model_2/lstm_1/lstm_cell_1/BiasAdd_3:output:0-model_2/lstm_1/lstm_cell_1/MatMul_7:product:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2"
 model_2/lstm_1/lstm_cell_1/add_4°
$model_2/lstm_1/lstm_cell_1/Sigmoid_2Sigmoid$model_2/lstm_1/lstm_cell_1/add_4:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2&
$model_2/lstm_1/lstm_cell_1/Sigmoid_2§
!model_2/lstm_1/lstm_cell_1/Tanh_1Tanh$model_2/lstm_1/lstm_cell_1/add_3:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2#
!model_2/lstm_1/lstm_cell_1/Tanh_1Ï
 model_2/lstm_1/lstm_cell_1/mul_6Mul(model_2/lstm_1/lstm_cell_1/Sigmoid_2:y:0%model_2/lstm_1/lstm_cell_1/Tanh_1:y:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2"
 model_2/lstm_1/lstm_cell_1/mul_6­
,model_2/lstm_1/TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ   2.
,model_2/lstm_1/TensorArrayV2_1/element_shapeò
model_2/lstm_1/TensorArrayV2_1TensorListReserve5model_2/lstm_1/TensorArrayV2_1/element_shape:output:0%model_2/lstm_1/strided_slice:output:0*
_output_shapes
: *
element_dtype0*

shape_type02 
model_2/lstm_1/TensorArrayV2_1l
model_2/lstm_1/timeConst*
_output_shapes
: *
dtype0*
value	B : 2
model_2/lstm_1/time§
,model_2/lstm_1/TensorArrayV2_2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ2.
,model_2/lstm_1/TensorArrayV2_2/element_shapeò
model_2/lstm_1/TensorArrayV2_2TensorListReserve5model_2/lstm_1/TensorArrayV2_2/element_shape:output:0%model_2/lstm_1/strided_slice:output:0*
_output_shapes
: *
element_dtype0
*

shape_type02 
model_2/lstm_1/TensorArrayV2_2á
Fmodel_2/lstm_1/TensorArrayUnstack_1/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ   2H
Fmodel_2/lstm_1/TensorArrayUnstack_1/TensorListFromTensor/element_shape¼
8model_2/lstm_1/TensorArrayUnstack_1/TensorListFromTensorTensorListFromTensormodel_2/lstm_1/transpose_1:y:0Omodel_2/lstm_1/TensorArrayUnstack_1/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0
*

shape_type02:
8model_2/lstm_1/TensorArrayUnstack_1/TensorListFromTensor
model_2/lstm_1/zeros_like	ZerosLike$model_2/lstm_1/lstm_cell_1/mul_6:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
model_2/lstm_1/zeros_like
'model_2/lstm_1/while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ2)
'model_2/lstm_1/while/maximum_iterations
!model_2/lstm_1/while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : 2#
!model_2/lstm_1/while/loop_counter«
model_2/lstm_1/whileWhile*model_2/lstm_1/while/loop_counter:output:00model_2/lstm_1/while/maximum_iterations:output:0model_2/lstm_1/time:output:0'model_2/lstm_1/TensorArrayV2_1:handle:0model_2/lstm_1/zeros_like:y:0input_4input_3%model_2/lstm_1/strided_slice:output:0Fmodel_2/lstm_1/TensorArrayUnstack/TensorListFromTensor:output_handle:0Hmodel_2/lstm_1/TensorArrayUnstack_1/TensorListFromTensor:output_handle:08model_2_lstm_1_lstm_cell_1_split_readvariableop_resource:model_2_lstm_1_lstm_cell_1_split_1_readvariableop_resource2model_2_lstm_1_lstm_cell_1_readvariableop_resource*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*d
_output_shapesR
P: : : : :ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ: : : : : : *%
_read_only_resource_inputs

*+
body#R!
model_2_lstm_1_while_body_13483*+
cond#R!
model_2_lstm_1_while_cond_13482*c
output_shapesR
P: : : : :ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ: : : : : : *
parallel_iterations 2
model_2/lstm_1/whileÓ
?model_2/lstm_1/TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ   2A
?model_2/lstm_1/TensorArrayV2Stack/TensorListStack/element_shape®
1model_2/lstm_1/TensorArrayV2Stack/TensorListStackTensorListStackmodel_2/lstm_1/while:output:3Hmodel_2/lstm_1/TensorArrayV2Stack/TensorListStack/element_shape:output:0*5
_output_shapes#
!:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ*
element_dtype023
1model_2/lstm_1/TensorArrayV2Stack/TensorListStack
$model_2/lstm_1/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB:
ÿÿÿÿÿÿÿÿÿ2&
$model_2/lstm_1/strided_slice_2/stack
&model_2/lstm_1/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2(
&model_2/lstm_1/strided_slice_2/stack_1
&model_2/lstm_1/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2(
&model_2/lstm_1/strided_slice_2/stack_2õ
model_2/lstm_1/strided_slice_2StridedSlice:model_2/lstm_1/TensorArrayV2Stack/TensorListStack:tensor:0-model_2/lstm_1/strided_slice_2/stack:output:0/model_2/lstm_1/strided_slice_2/stack_1:output:0/model_2/lstm_1/strided_slice_2/stack_2:output:0*
Index0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
shrink_axis_mask2 
model_2/lstm_1/strided_slice_2
model_2/lstm_1/transpose_2/permConst*
_output_shapes
:*
dtype0*!
valueB"          2!
model_2/lstm_1/transpose_2/permë
model_2/lstm_1/transpose_2	Transpose:model_2/lstm_1/TensorArrayV2Stack/TensorListStack:tensor:0(model_2/lstm_1/transpose_2/perm:output:0*
T0*5
_output_shapes#
!:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2
model_2/lstm_1/transpose_2
model_2/lstm_1/runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    2
model_2/lstm_1/runtimeÁ
&model_2/dense/Tensordot/ReadVariableOpReadVariableOp/model_2_dense_tensordot_readvariableop_resource*
_output_shapes
:	7*
dtype02(
&model_2/dense/Tensordot/ReadVariableOp
model_2/dense/Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:2
model_2/dense/Tensordot/axes
model_2/dense/Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       2
model_2/dense/Tensordot/free
model_2/dense/Tensordot/ShapeShapemodel_2/lstm_1/transpose_2:y:0*
T0*
_output_shapes
:2
model_2/dense/Tensordot/Shape
%model_2/dense/Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : 2'
%model_2/dense/Tensordot/GatherV2/axis
 model_2/dense/Tensordot/GatherV2GatherV2&model_2/dense/Tensordot/Shape:output:0%model_2/dense/Tensordot/free:output:0.model_2/dense/Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2"
 model_2/dense/Tensordot/GatherV2
'model_2/dense/Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2)
'model_2/dense/Tensordot/GatherV2_1/axis
"model_2/dense/Tensordot/GatherV2_1GatherV2&model_2/dense/Tensordot/Shape:output:0%model_2/dense/Tensordot/axes:output:00model_2/dense/Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2$
"model_2/dense/Tensordot/GatherV2_1
model_2/dense/Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2
model_2/dense/Tensordot/Const¸
model_2/dense/Tensordot/ProdProd)model_2/dense/Tensordot/GatherV2:output:0&model_2/dense/Tensordot/Const:output:0*
T0*
_output_shapes
: 2
model_2/dense/Tensordot/Prod
model_2/dense/Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: 2!
model_2/dense/Tensordot/Const_1À
model_2/dense/Tensordot/Prod_1Prod+model_2/dense/Tensordot/GatherV2_1:output:0(model_2/dense/Tensordot/Const_1:output:0*
T0*
_output_shapes
: 2 
model_2/dense/Tensordot/Prod_1
#model_2/dense/Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 2%
#model_2/dense/Tensordot/concat/axisö
model_2/dense/Tensordot/concatConcatV2%model_2/dense/Tensordot/free:output:0%model_2/dense/Tensordot/axes:output:0,model_2/dense/Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:2 
model_2/dense/Tensordot/concatÄ
model_2/dense/Tensordot/stackPack%model_2/dense/Tensordot/Prod:output:0'model_2/dense/Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:2
model_2/dense/Tensordot/stackÜ
!model_2/dense/Tensordot/transpose	Transposemodel_2/lstm_1/transpose_2:y:0'model_2/dense/Tensordot/concat:output:0*
T0*5
_output_shapes#
!:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2#
!model_2/dense/Tensordot/transpose×
model_2/dense/Tensordot/ReshapeReshape%model_2/dense/Tensordot/transpose:y:0&model_2/dense/Tensordot/stack:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2!
model_2/dense/Tensordot/ReshapeÖ
model_2/dense/Tensordot/MatMulMatMul(model_2/dense/Tensordot/Reshape:output:0.model_2/dense/Tensordot/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ72 
model_2/dense/Tensordot/MatMul
model_2/dense/Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:72!
model_2/dense/Tensordot/Const_2
%model_2/dense/Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2'
%model_2/dense/Tensordot/concat_1/axis
 model_2/dense/Tensordot/concat_1ConcatV2)model_2/dense/Tensordot/GatherV2:output:0(model_2/dense/Tensordot/Const_2:output:0.model_2/dense/Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:2"
 model_2/dense/Tensordot/concat_1Ñ
model_2/dense/TensordotReshape(model_2/dense/Tensordot/MatMul:product:0)model_2/dense/Tensordot/concat_1:output:0*
T0*4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ72
model_2/dense/Tensordot¶
$model_2/dense/BiasAdd/ReadVariableOpReadVariableOp-model_2_dense_biasadd_readvariableop_resource*
_output_shapes
:7*
dtype02&
$model_2/dense/BiasAdd/ReadVariableOpÈ
model_2/dense/BiasAddBiasAdd model_2/dense/Tensordot:output:0,model_2/dense/BiasAdd/ReadVariableOp:value:0*
T0*4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ72
model_2/dense/BiasAdd
#model_2/dense/Max/reduction_indicesConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ2%
#model_2/dense/Max/reduction_indicesË
model_2/dense/MaxMaxmodel_2/dense/BiasAdd:output:0,model_2/dense/Max/reduction_indices:output:0*
T0*4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ*
	keep_dims(2
model_2/dense/Max¨
model_2/dense/subSubmodel_2/dense/BiasAdd:output:0model_2/dense/Max:output:0*
T0*4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ72
model_2/dense/sub
model_2/dense/ExpExpmodel_2/dense/sub:z:0*
T0*4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ72
model_2/dense/Exp
#model_2/dense/Sum/reduction_indicesConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ2%
#model_2/dense/Sum/reduction_indicesÂ
model_2/dense/SumSummodel_2/dense/Exp:y:0,model_2/dense/Sum/reduction_indices:output:0*
T0*4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ*
	keep_dims(2
model_2/dense/Sum«
model_2/dense/truedivRealDivmodel_2/dense/Exp:y:0model_2/dense/Sum:output:0*
T0*4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ72
model_2/dense/truediv¤
IdentityIdentitymodel_2/dense/truediv:z:0%^model_2/dense/BiasAdd/ReadVariableOp'^model_2/dense/Tensordot/ReadVariableOp%^model_2/embedding_1/embedding_lookup*^model_2/lstm_1/lstm_cell_1/ReadVariableOp,^model_2/lstm_1/lstm_cell_1/ReadVariableOp_1,^model_2/lstm_1/lstm_cell_1/ReadVariableOp_2,^model_2/lstm_1/lstm_cell_1/ReadVariableOp_30^model_2/lstm_1/lstm_cell_1/split/ReadVariableOp2^model_2/lstm_1/lstm_cell_1/split_1/ReadVariableOp^model_2/lstm_1/while*
T0*4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ72

Identity 

Identity_1Identitymodel_2/lstm_1/while:output:5%^model_2/dense/BiasAdd/ReadVariableOp'^model_2/dense/Tensordot/ReadVariableOp%^model_2/embedding_1/embedding_lookup*^model_2/lstm_1/lstm_cell_1/ReadVariableOp,^model_2/lstm_1/lstm_cell_1/ReadVariableOp_1,^model_2/lstm_1/lstm_cell_1/ReadVariableOp_2,^model_2/lstm_1/lstm_cell_1/ReadVariableOp_30^model_2/lstm_1/lstm_cell_1/split/ReadVariableOp2^model_2/lstm_1/lstm_cell_1/split_1/ReadVariableOp^model_2/lstm_1/while*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity_1 

Identity_2Identitymodel_2/lstm_1/while:output:6%^model_2/dense/BiasAdd/ReadVariableOp'^model_2/dense/Tensordot/ReadVariableOp%^model_2/embedding_1/embedding_lookup*^model_2/lstm_1/lstm_cell_1/ReadVariableOp,^model_2/lstm_1/lstm_cell_1/ReadVariableOp_1,^model_2/lstm_1/lstm_cell_1/ReadVariableOp_2,^model_2/lstm_1/lstm_cell_1/ReadVariableOp_30^model_2/lstm_1/lstm_cell_1/split/ReadVariableOp2^model_2/lstm_1/lstm_cell_1/split_1/ReadVariableOp^model_2/lstm_1/while*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity_2"
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0*o
_input_shapes^
\:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ::::::2L
$model_2/dense/BiasAdd/ReadVariableOp$model_2/dense/BiasAdd/ReadVariableOp2P
&model_2/dense/Tensordot/ReadVariableOp&model_2/dense/Tensordot/ReadVariableOp2L
$model_2/embedding_1/embedding_lookup$model_2/embedding_1/embedding_lookup2V
)model_2/lstm_1/lstm_cell_1/ReadVariableOp)model_2/lstm_1/lstm_cell_1/ReadVariableOp2Z
+model_2/lstm_1/lstm_cell_1/ReadVariableOp_1+model_2/lstm_1/lstm_cell_1/ReadVariableOp_12Z
+model_2/lstm_1/lstm_cell_1/ReadVariableOp_2+model_2/lstm_1/lstm_cell_1/ReadVariableOp_22Z
+model_2/lstm_1/lstm_cell_1/ReadVariableOp_3+model_2/lstm_1/lstm_cell_1/ReadVariableOp_32b
/model_2/lstm_1/lstm_cell_1/split/ReadVariableOp/model_2/lstm_1/lstm_cell_1/split/ReadVariableOp2f
1model_2/lstm_1/lstm_cell_1/split_1/ReadVariableOp1model_2/lstm_1/lstm_cell_1/split_1/ReadVariableOp2,
model_2/lstm_1/whilemodel_2/lstm_1/while:Y U
0
_output_shapes
:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
!
_user_specified_name	input_2:QM
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
!
_user_specified_name	input_4:QM
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
!
_user_specified_name	input_3
Ü	

while_cond_16785
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_placeholder_4
while_less_strided_slice3
/while_while_cond_16785___redundant_placeholder03
/while_while_cond_16785___redundant_placeholder13
/while_while_cond_16785___redundant_placeholder23
/while_while_cond_16785___redundant_placeholder33
/while_while_cond_16785___redundant_placeholder4
while_identity
n

while/LessLesswhile_placeholderwhile_less_strided_slice*
T0*
_output_shapes
: 2

while/Less]
while/IdentityIdentitywhile/Less:z:0*
T0
*
_output_shapes
: 2
while/Identity")
while_identitywhile/Identity:output:0*m
_input_shapes\
Z: : : : :ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ: :::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :.*
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ:.*
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ:.*
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ:

_output_shapes
: :

_output_shapes
::	

_output_shapes
:

ï
#__inference_signature_wrapper_15312
input_2
input_3
input_4
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
	unknown_4
identity

identity_1

identity_2¢StatefulPartitionedCallÐ
StatefulPartitionedCallStatefulPartitionedCallinput_2input_4input_3unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4*
Tin
2	*
Tout
2*
_collective_manager_ids
 *\
_output_shapesJ
H:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ7:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ*(
_read_only_resource_inputs

*-
config_proto

CPU

GPU 2J 8 *)
f$R"
 __inference__wrapped_model_136652
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ72

Identity

Identity_1Identity StatefulPartitionedCall:output:1^StatefulPartitionedCall*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity_1

Identity_2Identity StatefulPartitionedCall:output:2^StatefulPartitionedCall*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity_2"
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0*o
_input_shapes^
\:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ::::::22
StatefulPartitionedCallStatefulPartitionedCall:Y U
0
_output_shapes
:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
!
_user_specified_name	input_2:QM
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
!
_user_specified_name	input_3:QM
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
!
_user_specified_name	input_4
ÔÃ

A__inference_lstm_1_layer_call_and_return_conditional_losses_14799

inputs
mask

initial_state
initial_state_1-
)lstm_cell_1_split_readvariableop_resource/
+lstm_cell_1_split_1_readvariableop_resource'
#lstm_cell_1_readvariableop_resource
identity

identity_1

identity_2¢lstm_cell_1/ReadVariableOp¢lstm_cell_1/ReadVariableOp_1¢lstm_cell_1/ReadVariableOp_2¢lstm_cell_1/ReadVariableOp_3¢ lstm_cell_1/split/ReadVariableOp¢"lstm_cell_1/split_1/ReadVariableOp¢whileu
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
transpose/perm
	transpose	Transposeinputstranspose/perm:output:0*
T0*5
_output_shapes#
!:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2
	transposeK
ShapeShapetranspose:y:0*
T0*
_output_shapes
:2
Shapet
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice/stackx
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_1x
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_2â
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slicek
ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ2
ExpandDims/dim

ExpandDims
ExpandDimsmaskExpandDims/dim:output:0*
T0
*4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2

ExpandDimsy
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
transpose_1/perm
transpose_1	TransposeExpandDims:output:0transpose_1/perm:output:0*
T0
*4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2
transpose_1
TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ2
TensorArrayV2/element_shape°
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
TensorArrayV2¿
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ   27
5TensorArrayUnstack/TensorListFromTensor/element_shapeø
'TensorArrayUnstack/TensorListFromTensorTensorListFromTensortranspose:y:0>TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type02)
'TensorArrayUnstack/TensorListFromTensorx
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_1/stack|
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_1|
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_2ý
strided_slice_1StridedSlicetranspose:y:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
shrink_axis_mask2
strided_slice_1w
lstm_cell_1/ones_like/ShapeShapeinitial_state*
T0*
_output_shapes
:2
lstm_cell_1/ones_like/Shape
lstm_cell_1/ones_like/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ?2
lstm_cell_1/ones_like/Constµ
lstm_cell_1/ones_likeFill$lstm_cell_1/ones_like/Shape:output:0$lstm_cell_1/ones_like/Const:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
lstm_cell_1/ones_like{
lstm_cell_1/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *ä8?2
lstm_cell_1/dropout/Const°
lstm_cell_1/dropout/MulMullstm_cell_1/ones_like:output:0"lstm_cell_1/dropout/Const:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
lstm_cell_1/dropout/Mul
lstm_cell_1/dropout/ShapeShapelstm_cell_1/ones_like:output:0*
T0*
_output_shapes
:2
lstm_cell_1/dropout/Shapeø
0lstm_cell_1/dropout/random_uniform/RandomUniformRandomUniform"lstm_cell_1/dropout/Shape:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
dtype0*
seed±ÿå)*
seed2ÓÀß22
0lstm_cell_1/dropout/random_uniform/RandomUniform
"lstm_cell_1/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *ÍÌÌ=2$
"lstm_cell_1/dropout/GreaterEqual/yï
 lstm_cell_1/dropout/GreaterEqualGreaterEqual9lstm_cell_1/dropout/random_uniform/RandomUniform:output:0+lstm_cell_1/dropout/GreaterEqual/y:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2"
 lstm_cell_1/dropout/GreaterEqual¤
lstm_cell_1/dropout/CastCast$lstm_cell_1/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
lstm_cell_1/dropout/Cast«
lstm_cell_1/dropout/Mul_1Mullstm_cell_1/dropout/Mul:z:0lstm_cell_1/dropout/Cast:y:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
lstm_cell_1/dropout/Mul_1
lstm_cell_1/dropout_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *ä8?2
lstm_cell_1/dropout_1/Const¶
lstm_cell_1/dropout_1/MulMullstm_cell_1/ones_like:output:0$lstm_cell_1/dropout_1/Const:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
lstm_cell_1/dropout_1/Mul
lstm_cell_1/dropout_1/ShapeShapelstm_cell_1/ones_like:output:0*
T0*
_output_shapes
:2
lstm_cell_1/dropout_1/Shapeþ
2lstm_cell_1/dropout_1/random_uniform/RandomUniformRandomUniform$lstm_cell_1/dropout_1/Shape:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
dtype0*
seed±ÿå)*
seed2ËÍô24
2lstm_cell_1/dropout_1/random_uniform/RandomUniform
$lstm_cell_1/dropout_1/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *ÍÌÌ=2&
$lstm_cell_1/dropout_1/GreaterEqual/y÷
"lstm_cell_1/dropout_1/GreaterEqualGreaterEqual;lstm_cell_1/dropout_1/random_uniform/RandomUniform:output:0-lstm_cell_1/dropout_1/GreaterEqual/y:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2$
"lstm_cell_1/dropout_1/GreaterEqualª
lstm_cell_1/dropout_1/CastCast&lstm_cell_1/dropout_1/GreaterEqual:z:0*

DstT0*

SrcT0
*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
lstm_cell_1/dropout_1/Cast³
lstm_cell_1/dropout_1/Mul_1Mullstm_cell_1/dropout_1/Mul:z:0lstm_cell_1/dropout_1/Cast:y:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
lstm_cell_1/dropout_1/Mul_1
lstm_cell_1/dropout_2/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *ä8?2
lstm_cell_1/dropout_2/Const¶
lstm_cell_1/dropout_2/MulMullstm_cell_1/ones_like:output:0$lstm_cell_1/dropout_2/Const:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
lstm_cell_1/dropout_2/Mul
lstm_cell_1/dropout_2/ShapeShapelstm_cell_1/ones_like:output:0*
T0*
_output_shapes
:2
lstm_cell_1/dropout_2/Shapeþ
2lstm_cell_1/dropout_2/random_uniform/RandomUniformRandomUniform$lstm_cell_1/dropout_2/Shape:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
dtype0*
seed±ÿå)*
seed2Åì24
2lstm_cell_1/dropout_2/random_uniform/RandomUniform
$lstm_cell_1/dropout_2/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *ÍÌÌ=2&
$lstm_cell_1/dropout_2/GreaterEqual/y÷
"lstm_cell_1/dropout_2/GreaterEqualGreaterEqual;lstm_cell_1/dropout_2/random_uniform/RandomUniform:output:0-lstm_cell_1/dropout_2/GreaterEqual/y:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2$
"lstm_cell_1/dropout_2/GreaterEqualª
lstm_cell_1/dropout_2/CastCast&lstm_cell_1/dropout_2/GreaterEqual:z:0*

DstT0*

SrcT0
*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
lstm_cell_1/dropout_2/Cast³
lstm_cell_1/dropout_2/Mul_1Mullstm_cell_1/dropout_2/Mul:z:0lstm_cell_1/dropout_2/Cast:y:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
lstm_cell_1/dropout_2/Mul_1
lstm_cell_1/dropout_3/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *ä8?2
lstm_cell_1/dropout_3/Const¶
lstm_cell_1/dropout_3/MulMullstm_cell_1/ones_like:output:0$lstm_cell_1/dropout_3/Const:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
lstm_cell_1/dropout_3/Mul
lstm_cell_1/dropout_3/ShapeShapelstm_cell_1/ones_like:output:0*
T0*
_output_shapes
:2
lstm_cell_1/dropout_3/Shapeþ
2lstm_cell_1/dropout_3/random_uniform/RandomUniformRandomUniform$lstm_cell_1/dropout_3/Shape:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
dtype0*
seed±ÿå)*
seed2øÞÃ24
2lstm_cell_1/dropout_3/random_uniform/RandomUniform
$lstm_cell_1/dropout_3/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *ÍÌÌ=2&
$lstm_cell_1/dropout_3/GreaterEqual/y÷
"lstm_cell_1/dropout_3/GreaterEqualGreaterEqual;lstm_cell_1/dropout_3/random_uniform/RandomUniform:output:0-lstm_cell_1/dropout_3/GreaterEqual/y:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2$
"lstm_cell_1/dropout_3/GreaterEqualª
lstm_cell_1/dropout_3/CastCast&lstm_cell_1/dropout_3/GreaterEqual:z:0*

DstT0*

SrcT0
*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
lstm_cell_1/dropout_3/Cast³
lstm_cell_1/dropout_3/Mul_1Mullstm_cell_1/dropout_3/Mul:z:0lstm_cell_1/dropout_3/Cast:y:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
lstm_cell_1/dropout_3/Mul_1h
lstm_cell_1/ConstConst*
_output_shapes
: *
dtype0*
value	B :2
lstm_cell_1/Const|
lstm_cell_1/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2
lstm_cell_1/split/split_dim°
 lstm_cell_1/split/ReadVariableOpReadVariableOp)lstm_cell_1_split_readvariableop_resource* 
_output_shapes
:
*
dtype02"
 lstm_cell_1/split/ReadVariableOpß
lstm_cell_1/splitSplit$lstm_cell_1/split/split_dim:output:0(lstm_cell_1/split/ReadVariableOp:value:0*
T0*D
_output_shapes2
0:
:
:
:
*
	num_split2
lstm_cell_1/split
lstm_cell_1/MatMulMatMulstrided_slice_1:output:0lstm_cell_1/split:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
lstm_cell_1/MatMul
lstm_cell_1/MatMul_1MatMulstrided_slice_1:output:0lstm_cell_1/split:output:1*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
lstm_cell_1/MatMul_1
lstm_cell_1/MatMul_2MatMulstrided_slice_1:output:0lstm_cell_1/split:output:2*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
lstm_cell_1/MatMul_2
lstm_cell_1/MatMul_3MatMulstrided_slice_1:output:0lstm_cell_1/split:output:3*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
lstm_cell_1/MatMul_3l
lstm_cell_1/Const_1Const*
_output_shapes
: *
dtype0*
value	B :2
lstm_cell_1/Const_1
lstm_cell_1/split_1/split_dimConst*
_output_shapes
: *
dtype0*
value	B : 2
lstm_cell_1/split_1/split_dim±
"lstm_cell_1/split_1/ReadVariableOpReadVariableOp+lstm_cell_1_split_1_readvariableop_resource*
_output_shapes	
:*
dtype02$
"lstm_cell_1/split_1/ReadVariableOpÓ
lstm_cell_1/split_1Split&lstm_cell_1/split_1/split_dim:output:0*lstm_cell_1/split_1/ReadVariableOp:value:0*
T0*0
_output_shapes
::::*
	num_split2
lstm_cell_1/split_1¤
lstm_cell_1/BiasAddBiasAddlstm_cell_1/MatMul:product:0lstm_cell_1/split_1:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
lstm_cell_1/BiasAddª
lstm_cell_1/BiasAdd_1BiasAddlstm_cell_1/MatMul_1:product:0lstm_cell_1/split_1:output:1*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
lstm_cell_1/BiasAdd_1ª
lstm_cell_1/BiasAdd_2BiasAddlstm_cell_1/MatMul_2:product:0lstm_cell_1/split_1:output:2*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
lstm_cell_1/BiasAdd_2ª
lstm_cell_1/BiasAdd_3BiasAddlstm_cell_1/MatMul_3:product:0lstm_cell_1/split_1:output:3*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
lstm_cell_1/BiasAdd_3
lstm_cell_1/mulMulinitial_statelstm_cell_1/dropout/Mul_1:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
lstm_cell_1/mul
lstm_cell_1/mul_1Mulinitial_statelstm_cell_1/dropout_1/Mul_1:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
lstm_cell_1/mul_1
lstm_cell_1/mul_2Mulinitial_statelstm_cell_1/dropout_2/Mul_1:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
lstm_cell_1/mul_2
lstm_cell_1/mul_3Mulinitial_statelstm_cell_1/dropout_3/Mul_1:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
lstm_cell_1/mul_3
lstm_cell_1/ReadVariableOpReadVariableOp#lstm_cell_1_readvariableop_resource* 
_output_shapes
:
*
dtype02
lstm_cell_1/ReadVariableOp
lstm_cell_1/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB"        2!
lstm_cell_1/strided_slice/stack
!lstm_cell_1/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       2#
!lstm_cell_1/strided_slice/stack_1
!lstm_cell_1/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2#
!lstm_cell_1/strided_slice/stack_2Æ
lstm_cell_1/strided_sliceStridedSlice"lstm_cell_1/ReadVariableOp:value:0(lstm_cell_1/strided_slice/stack:output:0*lstm_cell_1/strided_slice/stack_1:output:0*lstm_cell_1/strided_slice/stack_2:output:0*
Index0*
T0* 
_output_shapes
:
*

begin_mask*
end_mask2
lstm_cell_1/strided_slice¢
lstm_cell_1/MatMul_4MatMullstm_cell_1/mul:z:0"lstm_cell_1/strided_slice:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
lstm_cell_1/MatMul_4
lstm_cell_1/addAddV2lstm_cell_1/BiasAdd:output:0lstm_cell_1/MatMul_4:product:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
lstm_cell_1/add}
lstm_cell_1/SigmoidSigmoidlstm_cell_1/add:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
lstm_cell_1/Sigmoid¢
lstm_cell_1/ReadVariableOp_1ReadVariableOp#lstm_cell_1_readvariableop_resource* 
_output_shapes
:
*
dtype02
lstm_cell_1/ReadVariableOp_1
!lstm_cell_1/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB"       2#
!lstm_cell_1/strided_slice_1/stack
#lstm_cell_1/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       2%
#lstm_cell_1/strided_slice_1/stack_1
#lstm_cell_1/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2%
#lstm_cell_1/strided_slice_1/stack_2Ò
lstm_cell_1/strided_slice_1StridedSlice$lstm_cell_1/ReadVariableOp_1:value:0*lstm_cell_1/strided_slice_1/stack:output:0,lstm_cell_1/strided_slice_1/stack_1:output:0,lstm_cell_1/strided_slice_1/stack_2:output:0*
Index0*
T0* 
_output_shapes
:
*

begin_mask*
end_mask2
lstm_cell_1/strided_slice_1¦
lstm_cell_1/MatMul_5MatMullstm_cell_1/mul_1:z:0$lstm_cell_1/strided_slice_1:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
lstm_cell_1/MatMul_5¢
lstm_cell_1/add_1AddV2lstm_cell_1/BiasAdd_1:output:0lstm_cell_1/MatMul_5:product:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
lstm_cell_1/add_1
lstm_cell_1/Sigmoid_1Sigmoidlstm_cell_1/add_1:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
lstm_cell_1/Sigmoid_1
lstm_cell_1/mul_4Mullstm_cell_1/Sigmoid_1:y:0initial_state_1*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
lstm_cell_1/mul_4¢
lstm_cell_1/ReadVariableOp_2ReadVariableOp#lstm_cell_1_readvariableop_resource* 
_output_shapes
:
*
dtype02
lstm_cell_1/ReadVariableOp_2
!lstm_cell_1/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB"       2#
!lstm_cell_1/strided_slice_2/stack
#lstm_cell_1/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       2%
#lstm_cell_1/strided_slice_2/stack_1
#lstm_cell_1/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2%
#lstm_cell_1/strided_slice_2/stack_2Ò
lstm_cell_1/strided_slice_2StridedSlice$lstm_cell_1/ReadVariableOp_2:value:0*lstm_cell_1/strided_slice_2/stack:output:0,lstm_cell_1/strided_slice_2/stack_1:output:0,lstm_cell_1/strided_slice_2/stack_2:output:0*
Index0*
T0* 
_output_shapes
:
*

begin_mask*
end_mask2
lstm_cell_1/strided_slice_2¦
lstm_cell_1/MatMul_6MatMullstm_cell_1/mul_2:z:0$lstm_cell_1/strided_slice_2:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
lstm_cell_1/MatMul_6¢
lstm_cell_1/add_2AddV2lstm_cell_1/BiasAdd_2:output:0lstm_cell_1/MatMul_6:product:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
lstm_cell_1/add_2v
lstm_cell_1/TanhTanhlstm_cell_1/add_2:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
lstm_cell_1/Tanh
lstm_cell_1/mul_5Mullstm_cell_1/Sigmoid:y:0lstm_cell_1/Tanh:y:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
lstm_cell_1/mul_5
lstm_cell_1/add_3AddV2lstm_cell_1/mul_4:z:0lstm_cell_1/mul_5:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
lstm_cell_1/add_3¢
lstm_cell_1/ReadVariableOp_3ReadVariableOp#lstm_cell_1_readvariableop_resource* 
_output_shapes
:
*
dtype02
lstm_cell_1/ReadVariableOp_3
!lstm_cell_1/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB"       2#
!lstm_cell_1/strided_slice_3/stack
#lstm_cell_1/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        2%
#lstm_cell_1/strided_slice_3/stack_1
#lstm_cell_1/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2%
#lstm_cell_1/strided_slice_3/stack_2Ò
lstm_cell_1/strided_slice_3StridedSlice$lstm_cell_1/ReadVariableOp_3:value:0*lstm_cell_1/strided_slice_3/stack:output:0,lstm_cell_1/strided_slice_3/stack_1:output:0,lstm_cell_1/strided_slice_3/stack_2:output:0*
Index0*
T0* 
_output_shapes
:
*

begin_mask*
end_mask2
lstm_cell_1/strided_slice_3¦
lstm_cell_1/MatMul_7MatMullstm_cell_1/mul_3:z:0$lstm_cell_1/strided_slice_3:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
lstm_cell_1/MatMul_7¢
lstm_cell_1/add_4AddV2lstm_cell_1/BiasAdd_3:output:0lstm_cell_1/MatMul_7:product:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
lstm_cell_1/add_4
lstm_cell_1/Sigmoid_2Sigmoidlstm_cell_1/add_4:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
lstm_cell_1/Sigmoid_2z
lstm_cell_1/Tanh_1Tanhlstm_cell_1/add_3:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
lstm_cell_1/Tanh_1
lstm_cell_1/mul_6Mullstm_cell_1/Sigmoid_2:y:0lstm_cell_1/Tanh_1:y:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
lstm_cell_1/mul_6
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ   2
TensorArrayV2_1/element_shape¶
TensorArrayV2_1TensorListReserve&TensorArrayV2_1/element_shape:output:0strided_slice:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
TensorArrayV2_1N
timeConst*
_output_shapes
: *
dtype0*
value	B : 2
time
TensorArrayV2_2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ2
TensorArrayV2_2/element_shape¶
TensorArrayV2_2TensorListReserve&TensorArrayV2_2/element_shape:output:0strided_slice:output:0*
_output_shapes
: *
element_dtype0
*

shape_type02
TensorArrayV2_2Ã
7TensorArrayUnstack_1/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ   29
7TensorArrayUnstack_1/TensorListFromTensor/element_shape
)TensorArrayUnstack_1/TensorListFromTensorTensorListFromTensortranspose_1:y:0@TensorArrayUnstack_1/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0
*

shape_type02+
)TensorArrayUnstack_1/TensorListFromTensoro

zeros_like	ZerosLikelstm_cell_1/mul_6:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

zeros_like
while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ2
while/maximum_iterationsj
while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : 2
while/loop_counterØ
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros_like:y:0initial_stateinitial_state_1strided_slice:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:09TensorArrayUnstack_1/TensorListFromTensor:output_handle:0)lstm_cell_1_split_readvariableop_resource+lstm_cell_1_split_1_readvariableop_resource#lstm_cell_1_readvariableop_resource*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*d
_output_shapesR
P: : : : :ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ: : : : : : *%
_read_only_resource_inputs

*
bodyR
while_body_14618*
condR
while_cond_14617*c
output_shapesR
P: : : : :ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ: : : : : : *
parallel_iterations 2
whileµ
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ   22
0TensorArrayV2Stack/TensorListStack/element_shapeò
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*5
_output_shapes#
!:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ*
element_dtype02$
"TensorArrayV2Stack/TensorListStack
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB:
ÿÿÿÿÿÿÿÿÿ2
strided_slice_2/stack|
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_2/stack_1|
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_2/stack_2
strided_slice_2StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
shrink_axis_mask2
strided_slice_2y
transpose_2/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
transpose_2/perm¯
transpose_2	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_2/perm:output:0*
T0*5
_output_shapes#
!:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2
transpose_2f
runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    2	
runtime»
IdentityIdentitytranspose_2:y:0^lstm_cell_1/ReadVariableOp^lstm_cell_1/ReadVariableOp_1^lstm_cell_1/ReadVariableOp_2^lstm_cell_1/ReadVariableOp_3!^lstm_cell_1/split/ReadVariableOp#^lstm_cell_1/split_1/ReadVariableOp^while*
T0*5
_output_shapes#
!:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2

Identity±

Identity_1Identitywhile:output:5^lstm_cell_1/ReadVariableOp^lstm_cell_1/ReadVariableOp_1^lstm_cell_1/ReadVariableOp_2^lstm_cell_1/ReadVariableOp_3!^lstm_cell_1/split/ReadVariableOp#^lstm_cell_1/split_1/ReadVariableOp^while*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity_1±

Identity_2Identitywhile:output:6^lstm_cell_1/ReadVariableOp^lstm_cell_1/ReadVariableOp_1^lstm_cell_1/ReadVariableOp_2^lstm_cell_1/ReadVariableOp_3!^lstm_cell_1/split/ReadVariableOp#^lstm_cell_1/split_1/ReadVariableOp^while*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity_2"
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0*
_input_shapess
q:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:::28
lstm_cell_1/ReadVariableOplstm_cell_1/ReadVariableOp2<
lstm_cell_1/ReadVariableOp_1lstm_cell_1/ReadVariableOp_12<
lstm_cell_1/ReadVariableOp_2lstm_cell_1/ReadVariableOp_22<
lstm_cell_1/ReadVariableOp_3lstm_cell_1/ReadVariableOp_32D
 lstm_cell_1/split/ReadVariableOp lstm_cell_1/split/ReadVariableOp2H
"lstm_cell_1/split_1/ReadVariableOp"lstm_cell_1/split_1/ReadVariableOp2
whilewhile:] Y
5
_output_shapes#
!:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs:VR
0
_output_shapes
:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ

_user_specified_namemask:WS
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
'
_user_specified_nameinitial_state:WS
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
'
_user_specified_nameinitial_state
Ö
b
)__inference_dropout_1_layer_call_fn_16066

inputs
identity¢StatefulPartitionedCallè
StatefulPartitionedCallStatefulPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *5
_output_shapes#
!:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *M
fHRF
D__inference_dropout_1_layer_call_and_return_conditional_losses_144662
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*5
_output_shapes#
!:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*4
_input_shapes#
!:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ22
StatefulPartitionedCallStatefulPartitionedCall:] Y
5
_output_shapes#
!:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
¤

A__inference_lstm_1_layer_call_and_return_conditional_losses_15051

inputs
mask

initial_state
initial_state_1-
)lstm_cell_1_split_readvariableop_resource/
+lstm_cell_1_split_1_readvariableop_resource'
#lstm_cell_1_readvariableop_resource
identity

identity_1

identity_2¢lstm_cell_1/ReadVariableOp¢lstm_cell_1/ReadVariableOp_1¢lstm_cell_1/ReadVariableOp_2¢lstm_cell_1/ReadVariableOp_3¢ lstm_cell_1/split/ReadVariableOp¢"lstm_cell_1/split_1/ReadVariableOp¢whileu
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
transpose/perm
	transpose	Transposeinputstranspose/perm:output:0*
T0*5
_output_shapes#
!:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2
	transposeK
ShapeShapetranspose:y:0*
T0*
_output_shapes
:2
Shapet
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice/stackx
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_1x
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_2â
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slicek
ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ2
ExpandDims/dim

ExpandDims
ExpandDimsmaskExpandDims/dim:output:0*
T0
*4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2

ExpandDimsy
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
transpose_1/perm
transpose_1	TransposeExpandDims:output:0transpose_1/perm:output:0*
T0
*4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2
transpose_1
TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ2
TensorArrayV2/element_shape°
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
TensorArrayV2¿
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ   27
5TensorArrayUnstack/TensorListFromTensor/element_shapeø
'TensorArrayUnstack/TensorListFromTensorTensorListFromTensortranspose:y:0>TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type02)
'TensorArrayUnstack/TensorListFromTensorx
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_1/stack|
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_1|
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_2ý
strided_slice_1StridedSlicetranspose:y:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
shrink_axis_mask2
strided_slice_1w
lstm_cell_1/ones_like/ShapeShapeinitial_state*
T0*
_output_shapes
:2
lstm_cell_1/ones_like/Shape
lstm_cell_1/ones_like/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ?2
lstm_cell_1/ones_like/Constµ
lstm_cell_1/ones_likeFill$lstm_cell_1/ones_like/Shape:output:0$lstm_cell_1/ones_like/Const:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
lstm_cell_1/ones_likeh
lstm_cell_1/ConstConst*
_output_shapes
: *
dtype0*
value	B :2
lstm_cell_1/Const|
lstm_cell_1/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2
lstm_cell_1/split/split_dim°
 lstm_cell_1/split/ReadVariableOpReadVariableOp)lstm_cell_1_split_readvariableop_resource* 
_output_shapes
:
*
dtype02"
 lstm_cell_1/split/ReadVariableOpß
lstm_cell_1/splitSplit$lstm_cell_1/split/split_dim:output:0(lstm_cell_1/split/ReadVariableOp:value:0*
T0*D
_output_shapes2
0:
:
:
:
*
	num_split2
lstm_cell_1/split
lstm_cell_1/MatMulMatMulstrided_slice_1:output:0lstm_cell_1/split:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
lstm_cell_1/MatMul
lstm_cell_1/MatMul_1MatMulstrided_slice_1:output:0lstm_cell_1/split:output:1*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
lstm_cell_1/MatMul_1
lstm_cell_1/MatMul_2MatMulstrided_slice_1:output:0lstm_cell_1/split:output:2*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
lstm_cell_1/MatMul_2
lstm_cell_1/MatMul_3MatMulstrided_slice_1:output:0lstm_cell_1/split:output:3*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
lstm_cell_1/MatMul_3l
lstm_cell_1/Const_1Const*
_output_shapes
: *
dtype0*
value	B :2
lstm_cell_1/Const_1
lstm_cell_1/split_1/split_dimConst*
_output_shapes
: *
dtype0*
value	B : 2
lstm_cell_1/split_1/split_dim±
"lstm_cell_1/split_1/ReadVariableOpReadVariableOp+lstm_cell_1_split_1_readvariableop_resource*
_output_shapes	
:*
dtype02$
"lstm_cell_1/split_1/ReadVariableOpÓ
lstm_cell_1/split_1Split&lstm_cell_1/split_1/split_dim:output:0*lstm_cell_1/split_1/ReadVariableOp:value:0*
T0*0
_output_shapes
::::*
	num_split2
lstm_cell_1/split_1¤
lstm_cell_1/BiasAddBiasAddlstm_cell_1/MatMul:product:0lstm_cell_1/split_1:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
lstm_cell_1/BiasAddª
lstm_cell_1/BiasAdd_1BiasAddlstm_cell_1/MatMul_1:product:0lstm_cell_1/split_1:output:1*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
lstm_cell_1/BiasAdd_1ª
lstm_cell_1/BiasAdd_2BiasAddlstm_cell_1/MatMul_2:product:0lstm_cell_1/split_1:output:2*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
lstm_cell_1/BiasAdd_2ª
lstm_cell_1/BiasAdd_3BiasAddlstm_cell_1/MatMul_3:product:0lstm_cell_1/split_1:output:3*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
lstm_cell_1/BiasAdd_3
lstm_cell_1/mulMulinitial_statelstm_cell_1/ones_like:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
lstm_cell_1/mul
lstm_cell_1/mul_1Mulinitial_statelstm_cell_1/ones_like:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
lstm_cell_1/mul_1
lstm_cell_1/mul_2Mulinitial_statelstm_cell_1/ones_like:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
lstm_cell_1/mul_2
lstm_cell_1/mul_3Mulinitial_statelstm_cell_1/ones_like:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
lstm_cell_1/mul_3
lstm_cell_1/ReadVariableOpReadVariableOp#lstm_cell_1_readvariableop_resource* 
_output_shapes
:
*
dtype02
lstm_cell_1/ReadVariableOp
lstm_cell_1/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB"        2!
lstm_cell_1/strided_slice/stack
!lstm_cell_1/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       2#
!lstm_cell_1/strided_slice/stack_1
!lstm_cell_1/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2#
!lstm_cell_1/strided_slice/stack_2Æ
lstm_cell_1/strided_sliceStridedSlice"lstm_cell_1/ReadVariableOp:value:0(lstm_cell_1/strided_slice/stack:output:0*lstm_cell_1/strided_slice/stack_1:output:0*lstm_cell_1/strided_slice/stack_2:output:0*
Index0*
T0* 
_output_shapes
:
*

begin_mask*
end_mask2
lstm_cell_1/strided_slice¢
lstm_cell_1/MatMul_4MatMullstm_cell_1/mul:z:0"lstm_cell_1/strided_slice:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
lstm_cell_1/MatMul_4
lstm_cell_1/addAddV2lstm_cell_1/BiasAdd:output:0lstm_cell_1/MatMul_4:product:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
lstm_cell_1/add}
lstm_cell_1/SigmoidSigmoidlstm_cell_1/add:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
lstm_cell_1/Sigmoid¢
lstm_cell_1/ReadVariableOp_1ReadVariableOp#lstm_cell_1_readvariableop_resource* 
_output_shapes
:
*
dtype02
lstm_cell_1/ReadVariableOp_1
!lstm_cell_1/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB"       2#
!lstm_cell_1/strided_slice_1/stack
#lstm_cell_1/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       2%
#lstm_cell_1/strided_slice_1/stack_1
#lstm_cell_1/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2%
#lstm_cell_1/strided_slice_1/stack_2Ò
lstm_cell_1/strided_slice_1StridedSlice$lstm_cell_1/ReadVariableOp_1:value:0*lstm_cell_1/strided_slice_1/stack:output:0,lstm_cell_1/strided_slice_1/stack_1:output:0,lstm_cell_1/strided_slice_1/stack_2:output:0*
Index0*
T0* 
_output_shapes
:
*

begin_mask*
end_mask2
lstm_cell_1/strided_slice_1¦
lstm_cell_1/MatMul_5MatMullstm_cell_1/mul_1:z:0$lstm_cell_1/strided_slice_1:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
lstm_cell_1/MatMul_5¢
lstm_cell_1/add_1AddV2lstm_cell_1/BiasAdd_1:output:0lstm_cell_1/MatMul_5:product:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
lstm_cell_1/add_1
lstm_cell_1/Sigmoid_1Sigmoidlstm_cell_1/add_1:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
lstm_cell_1/Sigmoid_1
lstm_cell_1/mul_4Mullstm_cell_1/Sigmoid_1:y:0initial_state_1*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
lstm_cell_1/mul_4¢
lstm_cell_1/ReadVariableOp_2ReadVariableOp#lstm_cell_1_readvariableop_resource* 
_output_shapes
:
*
dtype02
lstm_cell_1/ReadVariableOp_2
!lstm_cell_1/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB"       2#
!lstm_cell_1/strided_slice_2/stack
#lstm_cell_1/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       2%
#lstm_cell_1/strided_slice_2/stack_1
#lstm_cell_1/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2%
#lstm_cell_1/strided_slice_2/stack_2Ò
lstm_cell_1/strided_slice_2StridedSlice$lstm_cell_1/ReadVariableOp_2:value:0*lstm_cell_1/strided_slice_2/stack:output:0,lstm_cell_1/strided_slice_2/stack_1:output:0,lstm_cell_1/strided_slice_2/stack_2:output:0*
Index0*
T0* 
_output_shapes
:
*

begin_mask*
end_mask2
lstm_cell_1/strided_slice_2¦
lstm_cell_1/MatMul_6MatMullstm_cell_1/mul_2:z:0$lstm_cell_1/strided_slice_2:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
lstm_cell_1/MatMul_6¢
lstm_cell_1/add_2AddV2lstm_cell_1/BiasAdd_2:output:0lstm_cell_1/MatMul_6:product:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
lstm_cell_1/add_2v
lstm_cell_1/TanhTanhlstm_cell_1/add_2:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
lstm_cell_1/Tanh
lstm_cell_1/mul_5Mullstm_cell_1/Sigmoid:y:0lstm_cell_1/Tanh:y:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
lstm_cell_1/mul_5
lstm_cell_1/add_3AddV2lstm_cell_1/mul_4:z:0lstm_cell_1/mul_5:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
lstm_cell_1/add_3¢
lstm_cell_1/ReadVariableOp_3ReadVariableOp#lstm_cell_1_readvariableop_resource* 
_output_shapes
:
*
dtype02
lstm_cell_1/ReadVariableOp_3
!lstm_cell_1/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB"       2#
!lstm_cell_1/strided_slice_3/stack
#lstm_cell_1/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        2%
#lstm_cell_1/strided_slice_3/stack_1
#lstm_cell_1/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2%
#lstm_cell_1/strided_slice_3/stack_2Ò
lstm_cell_1/strided_slice_3StridedSlice$lstm_cell_1/ReadVariableOp_3:value:0*lstm_cell_1/strided_slice_3/stack:output:0,lstm_cell_1/strided_slice_3/stack_1:output:0,lstm_cell_1/strided_slice_3/stack_2:output:0*
Index0*
T0* 
_output_shapes
:
*

begin_mask*
end_mask2
lstm_cell_1/strided_slice_3¦
lstm_cell_1/MatMul_7MatMullstm_cell_1/mul_3:z:0$lstm_cell_1/strided_slice_3:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
lstm_cell_1/MatMul_7¢
lstm_cell_1/add_4AddV2lstm_cell_1/BiasAdd_3:output:0lstm_cell_1/MatMul_7:product:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
lstm_cell_1/add_4
lstm_cell_1/Sigmoid_2Sigmoidlstm_cell_1/add_4:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
lstm_cell_1/Sigmoid_2z
lstm_cell_1/Tanh_1Tanhlstm_cell_1/add_3:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
lstm_cell_1/Tanh_1
lstm_cell_1/mul_6Mullstm_cell_1/Sigmoid_2:y:0lstm_cell_1/Tanh_1:y:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
lstm_cell_1/mul_6
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ   2
TensorArrayV2_1/element_shape¶
TensorArrayV2_1TensorListReserve&TensorArrayV2_1/element_shape:output:0strided_slice:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
TensorArrayV2_1N
timeConst*
_output_shapes
: *
dtype0*
value	B : 2
time
TensorArrayV2_2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ2
TensorArrayV2_2/element_shape¶
TensorArrayV2_2TensorListReserve&TensorArrayV2_2/element_shape:output:0strided_slice:output:0*
_output_shapes
: *
element_dtype0
*

shape_type02
TensorArrayV2_2Ã
7TensorArrayUnstack_1/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ   29
7TensorArrayUnstack_1/TensorListFromTensor/element_shape
)TensorArrayUnstack_1/TensorListFromTensorTensorListFromTensortranspose_1:y:0@TensorArrayUnstack_1/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0
*

shape_type02+
)TensorArrayUnstack_1/TensorListFromTensoro

zeros_like	ZerosLikelstm_cell_1/mul_6:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

zeros_like
while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ2
while/maximum_iterationsj
while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : 2
while/loop_counterØ
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros_like:y:0initial_stateinitial_state_1strided_slice:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:09TensorArrayUnstack_1/TensorListFromTensor:output_handle:0)lstm_cell_1_split_readvariableop_resource+lstm_cell_1_split_1_readvariableop_resource#lstm_cell_1_readvariableop_resource*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*d
_output_shapesR
P: : : : :ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ: : : : : : *%
_read_only_resource_inputs

*
bodyR
while_body_14902*
condR
while_cond_14901*c
output_shapesR
P: : : : :ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ: : : : : : *
parallel_iterations 2
whileµ
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ   22
0TensorArrayV2Stack/TensorListStack/element_shapeò
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*5
_output_shapes#
!:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ*
element_dtype02$
"TensorArrayV2Stack/TensorListStack
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB:
ÿÿÿÿÿÿÿÿÿ2
strided_slice_2/stack|
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_2/stack_1|
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_2/stack_2
strided_slice_2StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
shrink_axis_mask2
strided_slice_2y
transpose_2/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
transpose_2/perm¯
transpose_2	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_2/perm:output:0*
T0*5
_output_shapes#
!:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2
transpose_2f
runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    2	
runtime»
IdentityIdentitytranspose_2:y:0^lstm_cell_1/ReadVariableOp^lstm_cell_1/ReadVariableOp_1^lstm_cell_1/ReadVariableOp_2^lstm_cell_1/ReadVariableOp_3!^lstm_cell_1/split/ReadVariableOp#^lstm_cell_1/split_1/ReadVariableOp^while*
T0*5
_output_shapes#
!:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2

Identity±

Identity_1Identitywhile:output:5^lstm_cell_1/ReadVariableOp^lstm_cell_1/ReadVariableOp_1^lstm_cell_1/ReadVariableOp_2^lstm_cell_1/ReadVariableOp_3!^lstm_cell_1/split/ReadVariableOp#^lstm_cell_1/split_1/ReadVariableOp^while*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity_1±

Identity_2Identitywhile:output:6^lstm_cell_1/ReadVariableOp^lstm_cell_1/ReadVariableOp_1^lstm_cell_1/ReadVariableOp_2^lstm_cell_1/ReadVariableOp_3!^lstm_cell_1/split/ReadVariableOp#^lstm_cell_1/split_1/ReadVariableOp^while*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity_2"
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0*
_input_shapess
q:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:::28
lstm_cell_1/ReadVariableOplstm_cell_1/ReadVariableOp2<
lstm_cell_1/ReadVariableOp_1lstm_cell_1/ReadVariableOp_12<
lstm_cell_1/ReadVariableOp_2lstm_cell_1/ReadVariableOp_22<
lstm_cell_1/ReadVariableOp_3lstm_cell_1/ReadVariableOp_32D
 lstm_cell_1/split/ReadVariableOp lstm_cell_1/split/ReadVariableOp2H
"lstm_cell_1/split_1/ReadVariableOp"lstm_cell_1/split_1/ReadVariableOp2
whilewhile:] Y
5
_output_shapes#
!:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs:VR
0
_output_shapes
:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ

_user_specified_namemask:WS
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
'
_user_specified_nameinitial_state:WS
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
'
_user_specified_nameinitial_state
ál
þ
F__inference_lstm_cell_1_layer_call_and_return_conditional_losses_13814

inputs

states
states_1!
split_readvariableop_resource#
split_1_readvariableop_resource
readvariableop_resource
identity

identity_1

identity_2¢ReadVariableOp¢ReadVariableOp_1¢ReadVariableOp_2¢ReadVariableOp_3¢split/ReadVariableOp¢split_1/ReadVariableOpX
ones_like/ShapeShapestates*
T0*
_output_shapes
:2
ones_like/Shapeg
ones_like/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ?2
ones_like/Const
	ones_likeFillones_like/Shape:output:0ones_like/Const:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
	ones_likec
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *ä8?2
dropout/Const
dropout/MulMulones_like:output:0dropout/Const:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
dropout/Mul`
dropout/ShapeShapeones_like:output:0*
T0*
_output_shapes
:2
dropout/ShapeÔ
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
dtype0*
seed±ÿå)*
seed2À¿ü2&
$dropout/random_uniform/RandomUniformu
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *ÍÌÌ=2
dropout/GreaterEqual/y¿
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
dropout/GreaterEqual
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
dropout/Cast{
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
dropout/Mul_1g
dropout_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *ä8?2
dropout_1/Const
dropout_1/MulMulones_like:output:0dropout_1/Const:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
dropout_1/Muld
dropout_1/ShapeShapeones_like:output:0*
T0*
_output_shapes
:2
dropout_1/ShapeÚ
&dropout_1/random_uniform/RandomUniformRandomUniformdropout_1/Shape:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
dtype0*
seed±ÿå)*
seed2ö2(
&dropout_1/random_uniform/RandomUniformy
dropout_1/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *ÍÌÌ=2
dropout_1/GreaterEqual/yÇ
dropout_1/GreaterEqualGreaterEqual/dropout_1/random_uniform/RandomUniform:output:0!dropout_1/GreaterEqual/y:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
dropout_1/GreaterEqual
dropout_1/CastCastdropout_1/GreaterEqual:z:0*

DstT0*

SrcT0
*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
dropout_1/Cast
dropout_1/Mul_1Muldropout_1/Mul:z:0dropout_1/Cast:y:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
dropout_1/Mul_1g
dropout_2/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *ä8?2
dropout_2/Const
dropout_2/MulMulones_like:output:0dropout_2/Const:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
dropout_2/Muld
dropout_2/ShapeShapeones_like:output:0*
T0*
_output_shapes
:2
dropout_2/ShapeÚ
&dropout_2/random_uniform/RandomUniformRandomUniformdropout_2/Shape:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
dtype0*
seed±ÿå)*
seed2±Æ¾2(
&dropout_2/random_uniform/RandomUniformy
dropout_2/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *ÍÌÌ=2
dropout_2/GreaterEqual/yÇ
dropout_2/GreaterEqualGreaterEqual/dropout_2/random_uniform/RandomUniform:output:0!dropout_2/GreaterEqual/y:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
dropout_2/GreaterEqual
dropout_2/CastCastdropout_2/GreaterEqual:z:0*

DstT0*

SrcT0
*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
dropout_2/Cast
dropout_2/Mul_1Muldropout_2/Mul:z:0dropout_2/Cast:y:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
dropout_2/Mul_1g
dropout_3/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *ä8?2
dropout_3/Const
dropout_3/MulMulones_like:output:0dropout_3/Const:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
dropout_3/Muld
dropout_3/ShapeShapeones_like:output:0*
T0*
_output_shapes
:2
dropout_3/ShapeÚ
&dropout_3/random_uniform/RandomUniformRandomUniformdropout_3/Shape:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
dtype0*
seed±ÿå)*
seed2ïù2(
&dropout_3/random_uniform/RandomUniformy
dropout_3/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *ÍÌÌ=2
dropout_3/GreaterEqual/yÇ
dropout_3/GreaterEqualGreaterEqual/dropout_3/random_uniform/RandomUniform:output:0!dropout_3/GreaterEqual/y:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
dropout_3/GreaterEqual
dropout_3/CastCastdropout_3/GreaterEqual:z:0*

DstT0*

SrcT0
*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
dropout_3/Cast
dropout_3/Mul_1Muldropout_3/Mul:z:0dropout_3/Cast:y:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
dropout_3/Mul_1P
ConstConst*
_output_shapes
: *
dtype0*
value	B :2
Constd
split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2
split/split_dim
split/ReadVariableOpReadVariableOpsplit_readvariableop_resource* 
_output_shapes
:
*
dtype02
split/ReadVariableOp¯
splitSplitsplit/split_dim:output:0split/ReadVariableOp:value:0*
T0*D
_output_shapes2
0:
:
:
:
*
	num_split2
splite
MatMulMatMulinputssplit:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
MatMuli
MatMul_1MatMulinputssplit:output:1*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

MatMul_1i
MatMul_2MatMulinputssplit:output:2*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

MatMul_2i
MatMul_3MatMulinputssplit:output:3*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

MatMul_3T
Const_1Const*
_output_shapes
: *
dtype0*
value	B :2	
Const_1h
split_1/split_dimConst*
_output_shapes
: *
dtype0*
value	B : 2
split_1/split_dim
split_1/ReadVariableOpReadVariableOpsplit_1_readvariableop_resource*
_output_shapes	
:*
dtype02
split_1/ReadVariableOp£
split_1Splitsplit_1/split_dim:output:0split_1/ReadVariableOp:value:0*
T0*0
_output_shapes
::::*
	num_split2	
split_1t
BiasAddBiasAddMatMul:product:0split_1:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2	
BiasAddz
	BiasAdd_1BiasAddMatMul_1:product:0split_1:output:1*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
	BiasAdd_1z
	BiasAdd_2BiasAddMatMul_2:product:0split_1:output:2*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
	BiasAdd_2z
	BiasAdd_3BiasAddMatMul_3:product:0split_1:output:3*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
	BiasAdd_3_
mulMulstatesdropout/Mul_1:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
mule
mul_1Mulstatesdropout_1/Mul_1:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
mul_1e
mul_2Mulstatesdropout_2/Mul_1:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
mul_2e
mul_3Mulstatesdropout_3/Mul_1:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
mul_3z
ReadVariableOpReadVariableOpreadvariableop_resource* 
_output_shapes
:
*
dtype02
ReadVariableOp{
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB"        2
strided_slice/stack
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       2
strided_slice/stack_1
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2
strided_slice/stack_2þ
strided_sliceStridedSliceReadVariableOp:value:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0* 
_output_shapes
:
*

begin_mask*
end_mask2
strided_slicer
MatMul_4MatMulmul:z:0strided_slice:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

MatMul_4l
addAddV2BiasAdd:output:0MatMul_4:product:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
addY
SigmoidSigmoidadd:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2	
Sigmoid~
ReadVariableOp_1ReadVariableOpreadvariableop_resource* 
_output_shapes
:
*
dtype02
ReadVariableOp_1
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB"       2
strided_slice_1/stack
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       2
strided_slice_1/stack_1
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2
strided_slice_1/stack_2
strided_slice_1StridedSliceReadVariableOp_1:value:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0* 
_output_shapes
:
*

begin_mask*
end_mask2
strided_slice_1v
MatMul_5MatMul	mul_1:z:0strided_slice_1:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

MatMul_5r
add_1AddV2BiasAdd_1:output:0MatMul_5:product:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
add_1_
	Sigmoid_1Sigmoid	add_1:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
	Sigmoid_1a
mul_4MulSigmoid_1:y:0states_1*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
mul_4~
ReadVariableOp_2ReadVariableOpreadvariableop_resource* 
_output_shapes
:
*
dtype02
ReadVariableOp_2
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB"       2
strided_slice_2/stack
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       2
strided_slice_2/stack_1
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2
strided_slice_2/stack_2
strided_slice_2StridedSliceReadVariableOp_2:value:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0* 
_output_shapes
:
*

begin_mask*
end_mask2
strided_slice_2v
MatMul_6MatMul	mul_2:z:0strided_slice_2:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

MatMul_6r
add_2AddV2BiasAdd_2:output:0MatMul_6:product:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
add_2R
TanhTanh	add_2:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
Tanh_
mul_5MulSigmoid:y:0Tanh:y:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
mul_5`
add_3AddV2	mul_4:z:0	mul_5:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
add_3~
ReadVariableOp_3ReadVariableOpreadvariableop_resource* 
_output_shapes
:
*
dtype02
ReadVariableOp_3
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB"       2
strided_slice_3/stack
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        2
strided_slice_3/stack_1
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2
strided_slice_3/stack_2
strided_slice_3StridedSliceReadVariableOp_3:value:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0* 
_output_shapes
:
*

begin_mask*
end_mask2
strided_slice_3v
MatMul_7MatMul	mul_3:z:0strided_slice_3:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

MatMul_7r
add_4AddV2BiasAdd_3:output:0MatMul_7:product:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
add_4_
	Sigmoid_2Sigmoid	add_4:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
	Sigmoid_2V
Tanh_1Tanh	add_3:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
Tanh_1c
mul_6MulSigmoid_2:y:0
Tanh_1:y:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
mul_6Ø
IdentityIdentity	mul_6:z:0^ReadVariableOp^ReadVariableOp_1^ReadVariableOp_2^ReadVariableOp_3^split/ReadVariableOp^split_1/ReadVariableOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

IdentityÜ

Identity_1Identity	mul_6:z:0^ReadVariableOp^ReadVariableOp_1^ReadVariableOp_2^ReadVariableOp_3^split/ReadVariableOp^split_1/ReadVariableOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity_1Ü

Identity_2Identity	add_3:z:0^ReadVariableOp^ReadVariableOp_1^ReadVariableOp_2^ReadVariableOp_3^split/ReadVariableOp^split_1/ReadVariableOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity_2"
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0*[
_input_shapesJ
H:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:::2 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_12$
ReadVariableOp_2ReadVariableOp_22$
ReadVariableOp_3ReadVariableOp_32,
split/ReadVariableOpsplit/ReadVariableOp20
split_1/ReadVariableOpsplit_1/ReadVariableOp:P L
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs:PL
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_namestates:PL
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_namestates
ª
¾
while_cond_14326
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_less_strided_slice_13
/while_while_cond_14326___redundant_placeholder03
/while_while_cond_14326___redundant_placeholder13
/while_while_cond_14326___redundant_placeholder23
/while_while_cond_14326___redundant_placeholder3
while_identity
p

while/LessLesswhile_placeholderwhile_less_strided_slice_1*
T0*
_output_shapes
: 2

while/Less]
while/IdentityIdentitywhile/Less:z:0*
T0
*
_output_shapes
: 2
while/Identity")
while_identitywhile/Identity:output:0*U
_input_shapesD
B: : : : :ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ: ::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :.*
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ:.*
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ:

_output_shapes
: :

_output_shapes
:
î
q
+__inference_embedding_1_layer_call_fn_16034

inputs
unknown
identity¢StatefulPartitionedCall÷
StatefulPartitionedCallStatefulPartitionedCallinputsunknown*
Tin
2*
Tout
2*
_collective_manager_ids
 *5
_output_shapes#
!:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ*#
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *O
fJRH
F__inference_embedding_1_layer_call_and_return_conditional_losses_144272
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*5
_output_shapes#
!:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*3
_input_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ:22
StatefulPartitionedCallStatefulPartitionedCall:X T
0
_output_shapes
:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
É¹
Å
while_body_16215
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_05
1while_lstm_cell_1_split_readvariableop_resource_07
3while_lstm_cell_1_split_1_readvariableop_resource_0/
+while_lstm_cell_1_readvariableop_resource_0
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_identity_5
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor3
/while_lstm_cell_1_split_readvariableop_resource5
1while_lstm_cell_1_split_1_readvariableop_resource-
)while_lstm_cell_1_readvariableop_resource¢ while/lstm_cell_1/ReadVariableOp¢"while/lstm_cell_1/ReadVariableOp_1¢"while/lstm_cell_1/ReadVariableOp_2¢"while/lstm_cell_1/ReadVariableOp_3¢&while/lstm_cell_1/split/ReadVariableOp¢(while/lstm_cell_1/split_1/ReadVariableOpÃ
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ   29
7while/TensorArrayV2Read/TensorListGetItem/element_shapeÔ
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
element_dtype02+
)while/TensorArrayV2Read/TensorListGetItem
!while/lstm_cell_1/ones_like/ShapeShapewhile_placeholder_2*
T0*
_output_shapes
:2#
!while/lstm_cell_1/ones_like/Shape
!while/lstm_cell_1/ones_like/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ?2#
!while/lstm_cell_1/ones_like/ConstÍ
while/lstm_cell_1/ones_likeFill*while/lstm_cell_1/ones_like/Shape:output:0*while/lstm_cell_1/ones_like/Const:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
while/lstm_cell_1/ones_like
while/lstm_cell_1/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *ä8?2!
while/lstm_cell_1/dropout/ConstÈ
while/lstm_cell_1/dropout/MulMul$while/lstm_cell_1/ones_like:output:0(while/lstm_cell_1/dropout/Const:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
while/lstm_cell_1/dropout/Mul
while/lstm_cell_1/dropout/ShapeShape$while/lstm_cell_1/ones_like:output:0*
T0*
_output_shapes
:2!
while/lstm_cell_1/dropout/Shape
6while/lstm_cell_1/dropout/random_uniform/RandomUniformRandomUniform(while/lstm_cell_1/dropout/Shape:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
dtype0*
seed±ÿå)*
seed2ÏØú28
6while/lstm_cell_1/dropout/random_uniform/RandomUniform
(while/lstm_cell_1/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *ÍÌÌ=2*
(while/lstm_cell_1/dropout/GreaterEqual/y
&while/lstm_cell_1/dropout/GreaterEqualGreaterEqual?while/lstm_cell_1/dropout/random_uniform/RandomUniform:output:01while/lstm_cell_1/dropout/GreaterEqual/y:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2(
&while/lstm_cell_1/dropout/GreaterEqual¶
while/lstm_cell_1/dropout/CastCast*while/lstm_cell_1/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2 
while/lstm_cell_1/dropout/CastÃ
while/lstm_cell_1/dropout/Mul_1Mul!while/lstm_cell_1/dropout/Mul:z:0"while/lstm_cell_1/dropout/Cast:y:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2!
while/lstm_cell_1/dropout/Mul_1
!while/lstm_cell_1/dropout_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *ä8?2#
!while/lstm_cell_1/dropout_1/ConstÎ
while/lstm_cell_1/dropout_1/MulMul$while/lstm_cell_1/ones_like:output:0*while/lstm_cell_1/dropout_1/Const:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2!
while/lstm_cell_1/dropout_1/Mul
!while/lstm_cell_1/dropout_1/ShapeShape$while/lstm_cell_1/ones_like:output:0*
T0*
_output_shapes
:2#
!while/lstm_cell_1/dropout_1/Shape
8while/lstm_cell_1/dropout_1/random_uniform/RandomUniformRandomUniform*while/lstm_cell_1/dropout_1/Shape:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
dtype0*
seed±ÿå)*
seed2¶Ð2:
8while/lstm_cell_1/dropout_1/random_uniform/RandomUniform
*while/lstm_cell_1/dropout_1/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *ÍÌÌ=2,
*while/lstm_cell_1/dropout_1/GreaterEqual/y
(while/lstm_cell_1/dropout_1/GreaterEqualGreaterEqualAwhile/lstm_cell_1/dropout_1/random_uniform/RandomUniform:output:03while/lstm_cell_1/dropout_1/GreaterEqual/y:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2*
(while/lstm_cell_1/dropout_1/GreaterEqual¼
 while/lstm_cell_1/dropout_1/CastCast,while/lstm_cell_1/dropout_1/GreaterEqual:z:0*

DstT0*

SrcT0
*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2"
 while/lstm_cell_1/dropout_1/CastË
!while/lstm_cell_1/dropout_1/Mul_1Mul#while/lstm_cell_1/dropout_1/Mul:z:0$while/lstm_cell_1/dropout_1/Cast:y:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2#
!while/lstm_cell_1/dropout_1/Mul_1
!while/lstm_cell_1/dropout_2/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *ä8?2#
!while/lstm_cell_1/dropout_2/ConstÎ
while/lstm_cell_1/dropout_2/MulMul$while/lstm_cell_1/ones_like:output:0*while/lstm_cell_1/dropout_2/Const:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2!
while/lstm_cell_1/dropout_2/Mul
!while/lstm_cell_1/dropout_2/ShapeShape$while/lstm_cell_1/ones_like:output:0*
T0*
_output_shapes
:2#
!while/lstm_cell_1/dropout_2/Shape
8while/lstm_cell_1/dropout_2/random_uniform/RandomUniformRandomUniform*while/lstm_cell_1/dropout_2/Shape:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
dtype0*
seed±ÿå)*
seed2û§2:
8while/lstm_cell_1/dropout_2/random_uniform/RandomUniform
*while/lstm_cell_1/dropout_2/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *ÍÌÌ=2,
*while/lstm_cell_1/dropout_2/GreaterEqual/y
(while/lstm_cell_1/dropout_2/GreaterEqualGreaterEqualAwhile/lstm_cell_1/dropout_2/random_uniform/RandomUniform:output:03while/lstm_cell_1/dropout_2/GreaterEqual/y:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2*
(while/lstm_cell_1/dropout_2/GreaterEqual¼
 while/lstm_cell_1/dropout_2/CastCast,while/lstm_cell_1/dropout_2/GreaterEqual:z:0*

DstT0*

SrcT0
*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2"
 while/lstm_cell_1/dropout_2/CastË
!while/lstm_cell_1/dropout_2/Mul_1Mul#while/lstm_cell_1/dropout_2/Mul:z:0$while/lstm_cell_1/dropout_2/Cast:y:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2#
!while/lstm_cell_1/dropout_2/Mul_1
!while/lstm_cell_1/dropout_3/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *ä8?2#
!while/lstm_cell_1/dropout_3/ConstÎ
while/lstm_cell_1/dropout_3/MulMul$while/lstm_cell_1/ones_like:output:0*while/lstm_cell_1/dropout_3/Const:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2!
while/lstm_cell_1/dropout_3/Mul
!while/lstm_cell_1/dropout_3/ShapeShape$while/lstm_cell_1/ones_like:output:0*
T0*
_output_shapes
:2#
!while/lstm_cell_1/dropout_3/Shape
8while/lstm_cell_1/dropout_3/random_uniform/RandomUniformRandomUniform*while/lstm_cell_1/dropout_3/Shape:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
dtype0*
seed±ÿå)*
seed2ò¢2:
8while/lstm_cell_1/dropout_3/random_uniform/RandomUniform
*while/lstm_cell_1/dropout_3/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *ÍÌÌ=2,
*while/lstm_cell_1/dropout_3/GreaterEqual/y
(while/lstm_cell_1/dropout_3/GreaterEqualGreaterEqualAwhile/lstm_cell_1/dropout_3/random_uniform/RandomUniform:output:03while/lstm_cell_1/dropout_3/GreaterEqual/y:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2*
(while/lstm_cell_1/dropout_3/GreaterEqual¼
 while/lstm_cell_1/dropout_3/CastCast,while/lstm_cell_1/dropout_3/GreaterEqual:z:0*

DstT0*

SrcT0
*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2"
 while/lstm_cell_1/dropout_3/CastË
!while/lstm_cell_1/dropout_3/Mul_1Mul#while/lstm_cell_1/dropout_3/Mul:z:0$while/lstm_cell_1/dropout_3/Cast:y:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2#
!while/lstm_cell_1/dropout_3/Mul_1t
while/lstm_cell_1/ConstConst*
_output_shapes
: *
dtype0*
value	B :2
while/lstm_cell_1/Const
!while/lstm_cell_1/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2#
!while/lstm_cell_1/split/split_dimÄ
&while/lstm_cell_1/split/ReadVariableOpReadVariableOp1while_lstm_cell_1_split_readvariableop_resource_0* 
_output_shapes
:
*
dtype02(
&while/lstm_cell_1/split/ReadVariableOp÷
while/lstm_cell_1/splitSplit*while/lstm_cell_1/split/split_dim:output:0.while/lstm_cell_1/split/ReadVariableOp:value:0*
T0*D
_output_shapes2
0:
:
:
:
*
	num_split2
while/lstm_cell_1/splitÅ
while/lstm_cell_1/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:0 while/lstm_cell_1/split:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
while/lstm_cell_1/MatMulÉ
while/lstm_cell_1/MatMul_1MatMul0while/TensorArrayV2Read/TensorListGetItem:item:0 while/lstm_cell_1/split:output:1*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
while/lstm_cell_1/MatMul_1É
while/lstm_cell_1/MatMul_2MatMul0while/TensorArrayV2Read/TensorListGetItem:item:0 while/lstm_cell_1/split:output:2*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
while/lstm_cell_1/MatMul_2É
while/lstm_cell_1/MatMul_3MatMul0while/TensorArrayV2Read/TensorListGetItem:item:0 while/lstm_cell_1/split:output:3*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
while/lstm_cell_1/MatMul_3x
while/lstm_cell_1/Const_1Const*
_output_shapes
: *
dtype0*
value	B :2
while/lstm_cell_1/Const_1
#while/lstm_cell_1/split_1/split_dimConst*
_output_shapes
: *
dtype0*
value	B : 2%
#while/lstm_cell_1/split_1/split_dimÅ
(while/lstm_cell_1/split_1/ReadVariableOpReadVariableOp3while_lstm_cell_1_split_1_readvariableop_resource_0*
_output_shapes	
:*
dtype02*
(while/lstm_cell_1/split_1/ReadVariableOpë
while/lstm_cell_1/split_1Split,while/lstm_cell_1/split_1/split_dim:output:00while/lstm_cell_1/split_1/ReadVariableOp:value:0*
T0*0
_output_shapes
::::*
	num_split2
while/lstm_cell_1/split_1¼
while/lstm_cell_1/BiasAddBiasAdd"while/lstm_cell_1/MatMul:product:0"while/lstm_cell_1/split_1:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
while/lstm_cell_1/BiasAddÂ
while/lstm_cell_1/BiasAdd_1BiasAdd$while/lstm_cell_1/MatMul_1:product:0"while/lstm_cell_1/split_1:output:1*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
while/lstm_cell_1/BiasAdd_1Â
while/lstm_cell_1/BiasAdd_2BiasAdd$while/lstm_cell_1/MatMul_2:product:0"while/lstm_cell_1/split_1:output:2*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
while/lstm_cell_1/BiasAdd_2Â
while/lstm_cell_1/BiasAdd_3BiasAdd$while/lstm_cell_1/MatMul_3:product:0"while/lstm_cell_1/split_1:output:3*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
while/lstm_cell_1/BiasAdd_3¢
while/lstm_cell_1/mulMulwhile_placeholder_2#while/lstm_cell_1/dropout/Mul_1:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
while/lstm_cell_1/mul¨
while/lstm_cell_1/mul_1Mulwhile_placeholder_2%while/lstm_cell_1/dropout_1/Mul_1:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
while/lstm_cell_1/mul_1¨
while/lstm_cell_1/mul_2Mulwhile_placeholder_2%while/lstm_cell_1/dropout_2/Mul_1:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
while/lstm_cell_1/mul_2¨
while/lstm_cell_1/mul_3Mulwhile_placeholder_2%while/lstm_cell_1/dropout_3/Mul_1:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
while/lstm_cell_1/mul_3²
 while/lstm_cell_1/ReadVariableOpReadVariableOp+while_lstm_cell_1_readvariableop_resource_0* 
_output_shapes
:
*
dtype02"
 while/lstm_cell_1/ReadVariableOp
%while/lstm_cell_1/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB"        2'
%while/lstm_cell_1/strided_slice/stack£
'while/lstm_cell_1/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       2)
'while/lstm_cell_1/strided_slice/stack_1£
'while/lstm_cell_1/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2)
'while/lstm_cell_1/strided_slice/stack_2ê
while/lstm_cell_1/strided_sliceStridedSlice(while/lstm_cell_1/ReadVariableOp:value:0.while/lstm_cell_1/strided_slice/stack:output:00while/lstm_cell_1/strided_slice/stack_1:output:00while/lstm_cell_1/strided_slice/stack_2:output:0*
Index0*
T0* 
_output_shapes
:
*

begin_mask*
end_mask2!
while/lstm_cell_1/strided_sliceº
while/lstm_cell_1/MatMul_4MatMulwhile/lstm_cell_1/mul:z:0(while/lstm_cell_1/strided_slice:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
while/lstm_cell_1/MatMul_4´
while/lstm_cell_1/addAddV2"while/lstm_cell_1/BiasAdd:output:0$while/lstm_cell_1/MatMul_4:product:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
while/lstm_cell_1/add
while/lstm_cell_1/SigmoidSigmoidwhile/lstm_cell_1/add:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
while/lstm_cell_1/Sigmoid¶
"while/lstm_cell_1/ReadVariableOp_1ReadVariableOp+while_lstm_cell_1_readvariableop_resource_0* 
_output_shapes
:
*
dtype02$
"while/lstm_cell_1/ReadVariableOp_1£
'while/lstm_cell_1/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB"       2)
'while/lstm_cell_1/strided_slice_1/stack§
)while/lstm_cell_1/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       2+
)while/lstm_cell_1/strided_slice_1/stack_1§
)while/lstm_cell_1/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2+
)while/lstm_cell_1/strided_slice_1/stack_2ö
!while/lstm_cell_1/strided_slice_1StridedSlice*while/lstm_cell_1/ReadVariableOp_1:value:00while/lstm_cell_1/strided_slice_1/stack:output:02while/lstm_cell_1/strided_slice_1/stack_1:output:02while/lstm_cell_1/strided_slice_1/stack_2:output:0*
Index0*
T0* 
_output_shapes
:
*

begin_mask*
end_mask2#
!while/lstm_cell_1/strided_slice_1¾
while/lstm_cell_1/MatMul_5MatMulwhile/lstm_cell_1/mul_1:z:0*while/lstm_cell_1/strided_slice_1:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
while/lstm_cell_1/MatMul_5º
while/lstm_cell_1/add_1AddV2$while/lstm_cell_1/BiasAdd_1:output:0$while/lstm_cell_1/MatMul_5:product:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
while/lstm_cell_1/add_1
while/lstm_cell_1/Sigmoid_1Sigmoidwhile/lstm_cell_1/add_1:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
while/lstm_cell_1/Sigmoid_1¢
while/lstm_cell_1/mul_4Mulwhile/lstm_cell_1/Sigmoid_1:y:0while_placeholder_3*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
while/lstm_cell_1/mul_4¶
"while/lstm_cell_1/ReadVariableOp_2ReadVariableOp+while_lstm_cell_1_readvariableop_resource_0* 
_output_shapes
:
*
dtype02$
"while/lstm_cell_1/ReadVariableOp_2£
'while/lstm_cell_1/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB"       2)
'while/lstm_cell_1/strided_slice_2/stack§
)while/lstm_cell_1/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       2+
)while/lstm_cell_1/strided_slice_2/stack_1§
)while/lstm_cell_1/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2+
)while/lstm_cell_1/strided_slice_2/stack_2ö
!while/lstm_cell_1/strided_slice_2StridedSlice*while/lstm_cell_1/ReadVariableOp_2:value:00while/lstm_cell_1/strided_slice_2/stack:output:02while/lstm_cell_1/strided_slice_2/stack_1:output:02while/lstm_cell_1/strided_slice_2/stack_2:output:0*
Index0*
T0* 
_output_shapes
:
*

begin_mask*
end_mask2#
!while/lstm_cell_1/strided_slice_2¾
while/lstm_cell_1/MatMul_6MatMulwhile/lstm_cell_1/mul_2:z:0*while/lstm_cell_1/strided_slice_2:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
while/lstm_cell_1/MatMul_6º
while/lstm_cell_1/add_2AddV2$while/lstm_cell_1/BiasAdd_2:output:0$while/lstm_cell_1/MatMul_6:product:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
while/lstm_cell_1/add_2
while/lstm_cell_1/TanhTanhwhile/lstm_cell_1/add_2:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
while/lstm_cell_1/Tanh§
while/lstm_cell_1/mul_5Mulwhile/lstm_cell_1/Sigmoid:y:0while/lstm_cell_1/Tanh:y:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
while/lstm_cell_1/mul_5¨
while/lstm_cell_1/add_3AddV2while/lstm_cell_1/mul_4:z:0while/lstm_cell_1/mul_5:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
while/lstm_cell_1/add_3¶
"while/lstm_cell_1/ReadVariableOp_3ReadVariableOp+while_lstm_cell_1_readvariableop_resource_0* 
_output_shapes
:
*
dtype02$
"while/lstm_cell_1/ReadVariableOp_3£
'while/lstm_cell_1/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB"       2)
'while/lstm_cell_1/strided_slice_3/stack§
)while/lstm_cell_1/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        2+
)while/lstm_cell_1/strided_slice_3/stack_1§
)while/lstm_cell_1/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2+
)while/lstm_cell_1/strided_slice_3/stack_2ö
!while/lstm_cell_1/strided_slice_3StridedSlice*while/lstm_cell_1/ReadVariableOp_3:value:00while/lstm_cell_1/strided_slice_3/stack:output:02while/lstm_cell_1/strided_slice_3/stack_1:output:02while/lstm_cell_1/strided_slice_3/stack_2:output:0*
Index0*
T0* 
_output_shapes
:
*

begin_mask*
end_mask2#
!while/lstm_cell_1/strided_slice_3¾
while/lstm_cell_1/MatMul_7MatMulwhile/lstm_cell_1/mul_3:z:0*while/lstm_cell_1/strided_slice_3:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
while/lstm_cell_1/MatMul_7º
while/lstm_cell_1/add_4AddV2$while/lstm_cell_1/BiasAdd_3:output:0$while/lstm_cell_1/MatMul_7:product:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
while/lstm_cell_1/add_4
while/lstm_cell_1/Sigmoid_2Sigmoidwhile/lstm_cell_1/add_4:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
while/lstm_cell_1/Sigmoid_2
while/lstm_cell_1/Tanh_1Tanhwhile/lstm_cell_1/add_3:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
while/lstm_cell_1/Tanh_1«
while/lstm_cell_1/mul_6Mulwhile/lstm_cell_1/Sigmoid_2:y:0while/lstm_cell_1/Tanh_1:y:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
while/lstm_cell_1/mul_6ß
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholderwhile/lstm_cell_1/mul_6:z:0*
_output_shapes
: *
element_dtype02,
*while/TensorArrayV2Write/TensorListSetItem\
while/add/yConst*
_output_shapes
: *
dtype0*
value	B :2
while/add/yi
	while/addAddV2while_placeholderwhile/add/y:output:0*
T0*
_output_shapes
: 2
	while/add`
while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :2
while/add_1/yv
while/add_1AddV2while_while_loop_counterwhile/add_1/y:output:0*
T0*
_output_shapes
: 2
while/add_1Ä
while/IdentityIdentitywhile/add_1:z:0!^while/lstm_cell_1/ReadVariableOp#^while/lstm_cell_1/ReadVariableOp_1#^while/lstm_cell_1/ReadVariableOp_2#^while/lstm_cell_1/ReadVariableOp_3'^while/lstm_cell_1/split/ReadVariableOp)^while/lstm_cell_1/split_1/ReadVariableOp*
T0*
_output_shapes
: 2
while/Identity×
while/Identity_1Identitywhile_while_maximum_iterations!^while/lstm_cell_1/ReadVariableOp#^while/lstm_cell_1/ReadVariableOp_1#^while/lstm_cell_1/ReadVariableOp_2#^while/lstm_cell_1/ReadVariableOp_3'^while/lstm_cell_1/split/ReadVariableOp)^while/lstm_cell_1/split_1/ReadVariableOp*
T0*
_output_shapes
: 2
while/Identity_1Æ
while/Identity_2Identitywhile/add:z:0!^while/lstm_cell_1/ReadVariableOp#^while/lstm_cell_1/ReadVariableOp_1#^while/lstm_cell_1/ReadVariableOp_2#^while/lstm_cell_1/ReadVariableOp_3'^while/lstm_cell_1/split/ReadVariableOp)^while/lstm_cell_1/split_1/ReadVariableOp*
T0*
_output_shapes
: 2
while/Identity_2ó
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0!^while/lstm_cell_1/ReadVariableOp#^while/lstm_cell_1/ReadVariableOp_1#^while/lstm_cell_1/ReadVariableOp_2#^while/lstm_cell_1/ReadVariableOp_3'^while/lstm_cell_1/split/ReadVariableOp)^while/lstm_cell_1/split_1/ReadVariableOp*
T0*
_output_shapes
: 2
while/Identity_3æ
while/Identity_4Identitywhile/lstm_cell_1/mul_6:z:0!^while/lstm_cell_1/ReadVariableOp#^while/lstm_cell_1/ReadVariableOp_1#^while/lstm_cell_1/ReadVariableOp_2#^while/lstm_cell_1/ReadVariableOp_3'^while/lstm_cell_1/split/ReadVariableOp)^while/lstm_cell_1/split_1/ReadVariableOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
while/Identity_4æ
while/Identity_5Identitywhile/lstm_cell_1/add_3:z:0!^while/lstm_cell_1/ReadVariableOp#^while/lstm_cell_1/ReadVariableOp_1#^while/lstm_cell_1/ReadVariableOp_2#^while/lstm_cell_1/ReadVariableOp_3'^while/lstm_cell_1/split/ReadVariableOp)^while/lstm_cell_1/split_1/ReadVariableOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
while/Identity_5")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"-
while_identity_5while/Identity_5:output:0"X
)while_lstm_cell_1_readvariableop_resource+while_lstm_cell_1_readvariableop_resource_0"h
1while_lstm_cell_1_split_1_readvariableop_resource3while_lstm_cell_1_split_1_readvariableop_resource_0"d
/while_lstm_cell_1_split_readvariableop_resource1while_lstm_cell_1_split_readvariableop_resource_0"0
while_strided_slice_1while_strided_slice_1_0"¨
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*S
_input_shapesB
@: : : : :ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ: : :::2D
 while/lstm_cell_1/ReadVariableOp while/lstm_cell_1/ReadVariableOp2H
"while/lstm_cell_1/ReadVariableOp_1"while/lstm_cell_1/ReadVariableOp_12H
"while/lstm_cell_1/ReadVariableOp_2"while/lstm_cell_1/ReadVariableOp_22H
"while/lstm_cell_1/ReadVariableOp_3"while/lstm_cell_1/ReadVariableOp_32P
&while/lstm_cell_1/split/ReadVariableOp&while/lstm_cell_1/split/ReadVariableOp2T
(while/lstm_cell_1/split_1/ReadVariableOp(while/lstm_cell_1/split_1/ReadVariableOp: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :.*
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ:.*
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ:

_output_shapes
: :

_output_shapes
: 
Ä
«
lstm_1_while_cond_15788*
&lstm_1_while_lstm_1_while_loop_counter0
,lstm_1_while_lstm_1_while_maximum_iterations
lstm_1_while_placeholder
lstm_1_while_placeholder_1
lstm_1_while_placeholder_2
lstm_1_while_placeholder_3
lstm_1_while_placeholder_4*
&lstm_1_while_less_lstm_1_strided_sliceA
=lstm_1_while_lstm_1_while_cond_15788___redundant_placeholder0A
=lstm_1_while_lstm_1_while_cond_15788___redundant_placeholder1A
=lstm_1_while_lstm_1_while_cond_15788___redundant_placeholder2A
=lstm_1_while_lstm_1_while_cond_15788___redundant_placeholder3A
=lstm_1_while_lstm_1_while_cond_15788___redundant_placeholder4
lstm_1_while_identity

lstm_1/while/LessLesslstm_1_while_placeholder&lstm_1_while_less_lstm_1_strided_slice*
T0*
_output_shapes
: 2
lstm_1/while/Lessr
lstm_1/while/IdentityIdentitylstm_1/while/Less:z:0*
T0
*
_output_shapes
: 2
lstm_1/while/Identity"7
lstm_1_while_identitylstm_1/while/Identity:output:0*m
_input_shapes\
Z: : : : :ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ: :::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :.*
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ:.*
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ:.*
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ:

_output_shapes
: :

_output_shapes
::	

_output_shapes
:
ó
c
D__inference_dropout_1_layer_call_and_return_conditional_losses_16056

inputs
identityc
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *nÛ¶?2
dropout/Const
dropout/MulMulinputsdropout/Const:output:0*
T0*5
_output_shapes#
!:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2
dropout/MulT
dropout/ShapeShapeinputs*
T0*
_output_shapes
:2
dropout/ShapeÂ
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*5
_output_shapes#
!:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ*
dtype02&
$dropout/random_uniform/RandomUniformu
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *>2
dropout/GreaterEqual/yÌ
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*5
_output_shapes#
!:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2
dropout/GreaterEqual
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*5
_output_shapes#
!:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2
dropout/Cast
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*5
_output_shapes#
!:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2
dropout/Mul_1s
IdentityIdentitydropout/Mul_1:z:0*
T0*5
_output_shapes#
!:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*4
_input_shapes#
!:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ:] Y
5
_output_shapes#
!:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs



F__inference_embedding_1_layer_call_and_return_conditional_losses_16027

inputs
embedding_lookup_16021
identity¢embedding_lookupf
CastCastinputs*

DstT0*

SrcT0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2
Cast
embedding_lookupResourceGatherembedding_lookup_16021Cast:y:0",/job:localhost/replica:0/task:0/device:CPU:0*
Tindices0*)
_class
loc:@embedding_lookup/16021*5
_output_shapes#
!:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ*
dtype02
embedding_lookupö
embedding_lookup/IdentityIdentityembedding_lookup:output:0",/job:localhost/replica:0/task:0/device:CPU:0*
T0*)
_class
loc:@embedding_lookup/16021*5
_output_shapes#
!:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2
embedding_lookup/Identityª
embedding_lookup/Identity_1Identity"embedding_lookup/Identity:output:0*
T0*5
_output_shapes#
!:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2
embedding_lookup/Identity_1
IdentityIdentity$embedding_lookup/Identity_1:output:0^embedding_lookup*
T0*5
_output_shapes#
!:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*3
_input_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ:2$
embedding_lookupembedding_lookup:X T
0
_output_shapes
:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
Í
ã
model_2_lstm_1_while_cond_13482:
6model_2_lstm_1_while_model_2_lstm_1_while_loop_counter@
<model_2_lstm_1_while_model_2_lstm_1_while_maximum_iterations$
 model_2_lstm_1_while_placeholder&
"model_2_lstm_1_while_placeholder_1&
"model_2_lstm_1_while_placeholder_2&
"model_2_lstm_1_while_placeholder_3&
"model_2_lstm_1_while_placeholder_4:
6model_2_lstm_1_while_less_model_2_lstm_1_strided_sliceQ
Mmodel_2_lstm_1_while_model_2_lstm_1_while_cond_13482___redundant_placeholder0Q
Mmodel_2_lstm_1_while_model_2_lstm_1_while_cond_13482___redundant_placeholder1Q
Mmodel_2_lstm_1_while_model_2_lstm_1_while_cond_13482___redundant_placeholder2Q
Mmodel_2_lstm_1_while_model_2_lstm_1_while_cond_13482___redundant_placeholder3Q
Mmodel_2_lstm_1_while_model_2_lstm_1_while_cond_13482___redundant_placeholder4!
model_2_lstm_1_while_identity
¹
model_2/lstm_1/while/LessLess model_2_lstm_1_while_placeholder6model_2_lstm_1_while_less_model_2_lstm_1_strided_slice*
T0*
_output_shapes
: 2
model_2/lstm_1/while/Less
model_2/lstm_1/while/IdentityIdentitymodel_2/lstm_1/while/Less:z:0*
T0
*
_output_shapes
: 2
model_2/lstm_1/while/Identity"G
model_2_lstm_1_while_identity&model_2/lstm_1/while/Identity:output:0*m
_input_shapes\
Z: : : : :ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ: :::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :.*
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ:.*
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ:.*
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ:

_output_shapes
: :

_output_shapes
::	

_output_shapes
:
µì
Ç
lstm_1_while_body_15463*
&lstm_1_while_lstm_1_while_loop_counter0
,lstm_1_while_lstm_1_while_maximum_iterations
lstm_1_while_placeholder
lstm_1_while_placeholder_1
lstm_1_while_placeholder_2
lstm_1_while_placeholder_3
lstm_1_while_placeholder_4'
#lstm_1_while_lstm_1_strided_slice_0e
alstm_1_while_tensorarrayv2read_tensorlistgetitem_lstm_1_tensorarrayunstack_tensorlistfromtensor_0i
elstm_1_while_tensorarrayv2read_1_tensorlistgetitem_lstm_1_tensorarrayunstack_1_tensorlistfromtensor_0<
8lstm_1_while_lstm_cell_1_split_readvariableop_resource_0>
:lstm_1_while_lstm_cell_1_split_1_readvariableop_resource_06
2lstm_1_while_lstm_cell_1_readvariableop_resource_0
lstm_1_while_identity
lstm_1_while_identity_1
lstm_1_while_identity_2
lstm_1_while_identity_3
lstm_1_while_identity_4
lstm_1_while_identity_5
lstm_1_while_identity_6%
!lstm_1_while_lstm_1_strided_slicec
_lstm_1_while_tensorarrayv2read_tensorlistgetitem_lstm_1_tensorarrayunstack_tensorlistfromtensorg
clstm_1_while_tensorarrayv2read_1_tensorlistgetitem_lstm_1_tensorarrayunstack_1_tensorlistfromtensor:
6lstm_1_while_lstm_cell_1_split_readvariableop_resource<
8lstm_1_while_lstm_cell_1_split_1_readvariableop_resource4
0lstm_1_while_lstm_cell_1_readvariableop_resource¢'lstm_1/while/lstm_cell_1/ReadVariableOp¢)lstm_1/while/lstm_cell_1/ReadVariableOp_1¢)lstm_1/while/lstm_cell_1/ReadVariableOp_2¢)lstm_1/while/lstm_cell_1/ReadVariableOp_3¢-lstm_1/while/lstm_cell_1/split/ReadVariableOp¢/lstm_1/while/lstm_cell_1/split_1/ReadVariableOpÑ
>lstm_1/while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ   2@
>lstm_1/while/TensorArrayV2Read/TensorListGetItem/element_shapeþ
0lstm_1/while/TensorArrayV2Read/TensorListGetItemTensorListGetItemalstm_1_while_tensorarrayv2read_tensorlistgetitem_lstm_1_tensorarrayunstack_tensorlistfromtensor_0lstm_1_while_placeholderGlstm_1/while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
element_dtype022
0lstm_1/while/TensorArrayV2Read/TensorListGetItemÕ
@lstm_1/while/TensorArrayV2Read_1/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ   2B
@lstm_1/while/TensorArrayV2Read_1/TensorListGetItem/element_shape
2lstm_1/while/TensorArrayV2Read_1/TensorListGetItemTensorListGetItemelstm_1_while_tensorarrayv2read_1_tensorlistgetitem_lstm_1_tensorarrayunstack_1_tensorlistfromtensor_0lstm_1_while_placeholderIlstm_1/while/TensorArrayV2Read_1/TensorListGetItem/element_shape:output:0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
element_dtype0
24
2lstm_1/while/TensorArrayV2Read_1/TensorListGetItem
(lstm_1/while/lstm_cell_1/ones_like/ShapeShapelstm_1_while_placeholder_3*
T0*
_output_shapes
:2*
(lstm_1/while/lstm_cell_1/ones_like/Shape
(lstm_1/while/lstm_cell_1/ones_like/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ?2*
(lstm_1/while/lstm_cell_1/ones_like/Consté
"lstm_1/while/lstm_cell_1/ones_likeFill1lstm_1/while/lstm_cell_1/ones_like/Shape:output:01lstm_1/while/lstm_cell_1/ones_like/Const:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2$
"lstm_1/while/lstm_cell_1/ones_like
&lstm_1/while/lstm_cell_1/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *ä8?2(
&lstm_1/while/lstm_cell_1/dropout/Constä
$lstm_1/while/lstm_cell_1/dropout/MulMul+lstm_1/while/lstm_cell_1/ones_like:output:0/lstm_1/while/lstm_cell_1/dropout/Const:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2&
$lstm_1/while/lstm_cell_1/dropout/Mul«
&lstm_1/while/lstm_cell_1/dropout/ShapeShape+lstm_1/while/lstm_cell_1/ones_like:output:0*
T0*
_output_shapes
:2(
&lstm_1/while/lstm_cell_1/dropout/Shape
=lstm_1/while/lstm_cell_1/dropout/random_uniform/RandomUniformRandomUniform/lstm_1/while/lstm_cell_1/dropout/Shape:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
dtype0*
seed±ÿå)*
seed2Åýç2?
=lstm_1/while/lstm_cell_1/dropout/random_uniform/RandomUniform§
/lstm_1/while/lstm_cell_1/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *ÍÌÌ=21
/lstm_1/while/lstm_cell_1/dropout/GreaterEqual/y£
-lstm_1/while/lstm_cell_1/dropout/GreaterEqualGreaterEqualFlstm_1/while/lstm_cell_1/dropout/random_uniform/RandomUniform:output:08lstm_1/while/lstm_cell_1/dropout/GreaterEqual/y:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2/
-lstm_1/while/lstm_cell_1/dropout/GreaterEqualË
%lstm_1/while/lstm_cell_1/dropout/CastCast1lstm_1/while/lstm_cell_1/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2'
%lstm_1/while/lstm_cell_1/dropout/Castß
&lstm_1/while/lstm_cell_1/dropout/Mul_1Mul(lstm_1/while/lstm_cell_1/dropout/Mul:z:0)lstm_1/while/lstm_cell_1/dropout/Cast:y:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2(
&lstm_1/while/lstm_cell_1/dropout/Mul_1
(lstm_1/while/lstm_cell_1/dropout_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *ä8?2*
(lstm_1/while/lstm_cell_1/dropout_1/Constê
&lstm_1/while/lstm_cell_1/dropout_1/MulMul+lstm_1/while/lstm_cell_1/ones_like:output:01lstm_1/while/lstm_cell_1/dropout_1/Const:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2(
&lstm_1/while/lstm_cell_1/dropout_1/Mul¯
(lstm_1/while/lstm_cell_1/dropout_1/ShapeShape+lstm_1/while/lstm_cell_1/ones_like:output:0*
T0*
_output_shapes
:2*
(lstm_1/while/lstm_cell_1/dropout_1/Shape¥
?lstm_1/while/lstm_cell_1/dropout_1/random_uniform/RandomUniformRandomUniform1lstm_1/while/lstm_cell_1/dropout_1/Shape:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
dtype0*
seed±ÿå)*
seed2Éß2A
?lstm_1/while/lstm_cell_1/dropout_1/random_uniform/RandomUniform«
1lstm_1/while/lstm_cell_1/dropout_1/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *ÍÌÌ=23
1lstm_1/while/lstm_cell_1/dropout_1/GreaterEqual/y«
/lstm_1/while/lstm_cell_1/dropout_1/GreaterEqualGreaterEqualHlstm_1/while/lstm_cell_1/dropout_1/random_uniform/RandomUniform:output:0:lstm_1/while/lstm_cell_1/dropout_1/GreaterEqual/y:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ21
/lstm_1/while/lstm_cell_1/dropout_1/GreaterEqualÑ
'lstm_1/while/lstm_cell_1/dropout_1/CastCast3lstm_1/while/lstm_cell_1/dropout_1/GreaterEqual:z:0*

DstT0*

SrcT0
*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2)
'lstm_1/while/lstm_cell_1/dropout_1/Castç
(lstm_1/while/lstm_cell_1/dropout_1/Mul_1Mul*lstm_1/while/lstm_cell_1/dropout_1/Mul:z:0+lstm_1/while/lstm_cell_1/dropout_1/Cast:y:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2*
(lstm_1/while/lstm_cell_1/dropout_1/Mul_1
(lstm_1/while/lstm_cell_1/dropout_2/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *ä8?2*
(lstm_1/while/lstm_cell_1/dropout_2/Constê
&lstm_1/while/lstm_cell_1/dropout_2/MulMul+lstm_1/while/lstm_cell_1/ones_like:output:01lstm_1/while/lstm_cell_1/dropout_2/Const:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2(
&lstm_1/while/lstm_cell_1/dropout_2/Mul¯
(lstm_1/while/lstm_cell_1/dropout_2/ShapeShape+lstm_1/while/lstm_cell_1/ones_like:output:0*
T0*
_output_shapes
:2*
(lstm_1/while/lstm_cell_1/dropout_2/Shape¤
?lstm_1/while/lstm_cell_1/dropout_2/random_uniform/RandomUniformRandomUniform1lstm_1/while/lstm_cell_1/dropout_2/Shape:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
dtype0*
seed±ÿå)*
seed2ö¾2A
?lstm_1/while/lstm_cell_1/dropout_2/random_uniform/RandomUniform«
1lstm_1/while/lstm_cell_1/dropout_2/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *ÍÌÌ=23
1lstm_1/while/lstm_cell_1/dropout_2/GreaterEqual/y«
/lstm_1/while/lstm_cell_1/dropout_2/GreaterEqualGreaterEqualHlstm_1/while/lstm_cell_1/dropout_2/random_uniform/RandomUniform:output:0:lstm_1/while/lstm_cell_1/dropout_2/GreaterEqual/y:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ21
/lstm_1/while/lstm_cell_1/dropout_2/GreaterEqualÑ
'lstm_1/while/lstm_cell_1/dropout_2/CastCast3lstm_1/while/lstm_cell_1/dropout_2/GreaterEqual:z:0*

DstT0*

SrcT0
*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2)
'lstm_1/while/lstm_cell_1/dropout_2/Castç
(lstm_1/while/lstm_cell_1/dropout_2/Mul_1Mul*lstm_1/while/lstm_cell_1/dropout_2/Mul:z:0+lstm_1/while/lstm_cell_1/dropout_2/Cast:y:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2*
(lstm_1/while/lstm_cell_1/dropout_2/Mul_1
(lstm_1/while/lstm_cell_1/dropout_3/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *ä8?2*
(lstm_1/while/lstm_cell_1/dropout_3/Constê
&lstm_1/while/lstm_cell_1/dropout_3/MulMul+lstm_1/while/lstm_cell_1/ones_like:output:01lstm_1/while/lstm_cell_1/dropout_3/Const:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2(
&lstm_1/while/lstm_cell_1/dropout_3/Mul¯
(lstm_1/while/lstm_cell_1/dropout_3/ShapeShape+lstm_1/while/lstm_cell_1/ones_like:output:0*
T0*
_output_shapes
:2*
(lstm_1/while/lstm_cell_1/dropout_3/Shape¥
?lstm_1/while/lstm_cell_1/dropout_3/random_uniform/RandomUniformRandomUniform1lstm_1/while/lstm_cell_1/dropout_3/Shape:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
dtype0*
seed±ÿå)*
seed2¼¡2A
?lstm_1/while/lstm_cell_1/dropout_3/random_uniform/RandomUniform«
1lstm_1/while/lstm_cell_1/dropout_3/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *ÍÌÌ=23
1lstm_1/while/lstm_cell_1/dropout_3/GreaterEqual/y«
/lstm_1/while/lstm_cell_1/dropout_3/GreaterEqualGreaterEqualHlstm_1/while/lstm_cell_1/dropout_3/random_uniform/RandomUniform:output:0:lstm_1/while/lstm_cell_1/dropout_3/GreaterEqual/y:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ21
/lstm_1/while/lstm_cell_1/dropout_3/GreaterEqualÑ
'lstm_1/while/lstm_cell_1/dropout_3/CastCast3lstm_1/while/lstm_cell_1/dropout_3/GreaterEqual:z:0*

DstT0*

SrcT0
*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2)
'lstm_1/while/lstm_cell_1/dropout_3/Castç
(lstm_1/while/lstm_cell_1/dropout_3/Mul_1Mul*lstm_1/while/lstm_cell_1/dropout_3/Mul:z:0+lstm_1/while/lstm_cell_1/dropout_3/Cast:y:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2*
(lstm_1/while/lstm_cell_1/dropout_3/Mul_1
lstm_1/while/lstm_cell_1/ConstConst*
_output_shapes
: *
dtype0*
value	B :2 
lstm_1/while/lstm_cell_1/Const
(lstm_1/while/lstm_cell_1/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2*
(lstm_1/while/lstm_cell_1/split/split_dimÙ
-lstm_1/while/lstm_cell_1/split/ReadVariableOpReadVariableOp8lstm_1_while_lstm_cell_1_split_readvariableop_resource_0* 
_output_shapes
:
*
dtype02/
-lstm_1/while/lstm_cell_1/split/ReadVariableOp
lstm_1/while/lstm_cell_1/splitSplit1lstm_1/while/lstm_cell_1/split/split_dim:output:05lstm_1/while/lstm_cell_1/split/ReadVariableOp:value:0*
T0*D
_output_shapes2
0:
:
:
:
*
	num_split2 
lstm_1/while/lstm_cell_1/splitá
lstm_1/while/lstm_cell_1/MatMulMatMul7lstm_1/while/TensorArrayV2Read/TensorListGetItem:item:0'lstm_1/while/lstm_cell_1/split:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2!
lstm_1/while/lstm_cell_1/MatMulå
!lstm_1/while/lstm_cell_1/MatMul_1MatMul7lstm_1/while/TensorArrayV2Read/TensorListGetItem:item:0'lstm_1/while/lstm_cell_1/split:output:1*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2#
!lstm_1/while/lstm_cell_1/MatMul_1å
!lstm_1/while/lstm_cell_1/MatMul_2MatMul7lstm_1/while/TensorArrayV2Read/TensorListGetItem:item:0'lstm_1/while/lstm_cell_1/split:output:2*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2#
!lstm_1/while/lstm_cell_1/MatMul_2å
!lstm_1/while/lstm_cell_1/MatMul_3MatMul7lstm_1/while/TensorArrayV2Read/TensorListGetItem:item:0'lstm_1/while/lstm_cell_1/split:output:3*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2#
!lstm_1/while/lstm_cell_1/MatMul_3
 lstm_1/while/lstm_cell_1/Const_1Const*
_output_shapes
: *
dtype0*
value	B :2"
 lstm_1/while/lstm_cell_1/Const_1
*lstm_1/while/lstm_cell_1/split_1/split_dimConst*
_output_shapes
: *
dtype0*
value	B : 2,
*lstm_1/while/lstm_cell_1/split_1/split_dimÚ
/lstm_1/while/lstm_cell_1/split_1/ReadVariableOpReadVariableOp:lstm_1_while_lstm_cell_1_split_1_readvariableop_resource_0*
_output_shapes	
:*
dtype021
/lstm_1/while/lstm_cell_1/split_1/ReadVariableOp
 lstm_1/while/lstm_cell_1/split_1Split3lstm_1/while/lstm_cell_1/split_1/split_dim:output:07lstm_1/while/lstm_cell_1/split_1/ReadVariableOp:value:0*
T0*0
_output_shapes
::::*
	num_split2"
 lstm_1/while/lstm_cell_1/split_1Ø
 lstm_1/while/lstm_cell_1/BiasAddBiasAdd)lstm_1/while/lstm_cell_1/MatMul:product:0)lstm_1/while/lstm_cell_1/split_1:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2"
 lstm_1/while/lstm_cell_1/BiasAddÞ
"lstm_1/while/lstm_cell_1/BiasAdd_1BiasAdd+lstm_1/while/lstm_cell_1/MatMul_1:product:0)lstm_1/while/lstm_cell_1/split_1:output:1*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2$
"lstm_1/while/lstm_cell_1/BiasAdd_1Þ
"lstm_1/while/lstm_cell_1/BiasAdd_2BiasAdd+lstm_1/while/lstm_cell_1/MatMul_2:product:0)lstm_1/while/lstm_cell_1/split_1:output:2*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2$
"lstm_1/while/lstm_cell_1/BiasAdd_2Þ
"lstm_1/while/lstm_cell_1/BiasAdd_3BiasAdd+lstm_1/while/lstm_cell_1/MatMul_3:product:0)lstm_1/while/lstm_cell_1/split_1:output:3*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2$
"lstm_1/while/lstm_cell_1/BiasAdd_3¾
lstm_1/while/lstm_cell_1/mulMullstm_1_while_placeholder_3*lstm_1/while/lstm_cell_1/dropout/Mul_1:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
lstm_1/while/lstm_cell_1/mulÄ
lstm_1/while/lstm_cell_1/mul_1Mullstm_1_while_placeholder_3,lstm_1/while/lstm_cell_1/dropout_1/Mul_1:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2 
lstm_1/while/lstm_cell_1/mul_1Ä
lstm_1/while/lstm_cell_1/mul_2Mullstm_1_while_placeholder_3,lstm_1/while/lstm_cell_1/dropout_2/Mul_1:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2 
lstm_1/while/lstm_cell_1/mul_2Ä
lstm_1/while/lstm_cell_1/mul_3Mullstm_1_while_placeholder_3,lstm_1/while/lstm_cell_1/dropout_3/Mul_1:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2 
lstm_1/while/lstm_cell_1/mul_3Ç
'lstm_1/while/lstm_cell_1/ReadVariableOpReadVariableOp2lstm_1_while_lstm_cell_1_readvariableop_resource_0* 
_output_shapes
:
*
dtype02)
'lstm_1/while/lstm_cell_1/ReadVariableOp­
,lstm_1/while/lstm_cell_1/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB"        2.
,lstm_1/while/lstm_cell_1/strided_slice/stack±
.lstm_1/while/lstm_cell_1/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       20
.lstm_1/while/lstm_cell_1/strided_slice/stack_1±
.lstm_1/while/lstm_cell_1/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      20
.lstm_1/while/lstm_cell_1/strided_slice/stack_2
&lstm_1/while/lstm_cell_1/strided_sliceStridedSlice/lstm_1/while/lstm_cell_1/ReadVariableOp:value:05lstm_1/while/lstm_cell_1/strided_slice/stack:output:07lstm_1/while/lstm_cell_1/strided_slice/stack_1:output:07lstm_1/while/lstm_cell_1/strided_slice/stack_2:output:0*
Index0*
T0* 
_output_shapes
:
*

begin_mask*
end_mask2(
&lstm_1/while/lstm_cell_1/strided_sliceÖ
!lstm_1/while/lstm_cell_1/MatMul_4MatMul lstm_1/while/lstm_cell_1/mul:z:0/lstm_1/while/lstm_cell_1/strided_slice:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2#
!lstm_1/while/lstm_cell_1/MatMul_4Ð
lstm_1/while/lstm_cell_1/addAddV2)lstm_1/while/lstm_cell_1/BiasAdd:output:0+lstm_1/while/lstm_cell_1/MatMul_4:product:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
lstm_1/while/lstm_cell_1/add¤
 lstm_1/while/lstm_cell_1/SigmoidSigmoid lstm_1/while/lstm_cell_1/add:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2"
 lstm_1/while/lstm_cell_1/SigmoidË
)lstm_1/while/lstm_cell_1/ReadVariableOp_1ReadVariableOp2lstm_1_while_lstm_cell_1_readvariableop_resource_0* 
_output_shapes
:
*
dtype02+
)lstm_1/while/lstm_cell_1/ReadVariableOp_1±
.lstm_1/while/lstm_cell_1/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB"       20
.lstm_1/while/lstm_cell_1/strided_slice_1/stackµ
0lstm_1/while/lstm_cell_1/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       22
0lstm_1/while/lstm_cell_1/strided_slice_1/stack_1µ
0lstm_1/while/lstm_cell_1/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      22
0lstm_1/while/lstm_cell_1/strided_slice_1/stack_2 
(lstm_1/while/lstm_cell_1/strided_slice_1StridedSlice1lstm_1/while/lstm_cell_1/ReadVariableOp_1:value:07lstm_1/while/lstm_cell_1/strided_slice_1/stack:output:09lstm_1/while/lstm_cell_1/strided_slice_1/stack_1:output:09lstm_1/while/lstm_cell_1/strided_slice_1/stack_2:output:0*
Index0*
T0* 
_output_shapes
:
*

begin_mask*
end_mask2*
(lstm_1/while/lstm_cell_1/strided_slice_1Ú
!lstm_1/while/lstm_cell_1/MatMul_5MatMul"lstm_1/while/lstm_cell_1/mul_1:z:01lstm_1/while/lstm_cell_1/strided_slice_1:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2#
!lstm_1/while/lstm_cell_1/MatMul_5Ö
lstm_1/while/lstm_cell_1/add_1AddV2+lstm_1/while/lstm_cell_1/BiasAdd_1:output:0+lstm_1/while/lstm_cell_1/MatMul_5:product:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2 
lstm_1/while/lstm_cell_1/add_1ª
"lstm_1/while/lstm_cell_1/Sigmoid_1Sigmoid"lstm_1/while/lstm_cell_1/add_1:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2$
"lstm_1/while/lstm_cell_1/Sigmoid_1¾
lstm_1/while/lstm_cell_1/mul_4Mul&lstm_1/while/lstm_cell_1/Sigmoid_1:y:0lstm_1_while_placeholder_4*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2 
lstm_1/while/lstm_cell_1/mul_4Ë
)lstm_1/while/lstm_cell_1/ReadVariableOp_2ReadVariableOp2lstm_1_while_lstm_cell_1_readvariableop_resource_0* 
_output_shapes
:
*
dtype02+
)lstm_1/while/lstm_cell_1/ReadVariableOp_2±
.lstm_1/while/lstm_cell_1/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB"       20
.lstm_1/while/lstm_cell_1/strided_slice_2/stackµ
0lstm_1/while/lstm_cell_1/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       22
0lstm_1/while/lstm_cell_1/strided_slice_2/stack_1µ
0lstm_1/while/lstm_cell_1/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      22
0lstm_1/while/lstm_cell_1/strided_slice_2/stack_2 
(lstm_1/while/lstm_cell_1/strided_slice_2StridedSlice1lstm_1/while/lstm_cell_1/ReadVariableOp_2:value:07lstm_1/while/lstm_cell_1/strided_slice_2/stack:output:09lstm_1/while/lstm_cell_1/strided_slice_2/stack_1:output:09lstm_1/while/lstm_cell_1/strided_slice_2/stack_2:output:0*
Index0*
T0* 
_output_shapes
:
*

begin_mask*
end_mask2*
(lstm_1/while/lstm_cell_1/strided_slice_2Ú
!lstm_1/while/lstm_cell_1/MatMul_6MatMul"lstm_1/while/lstm_cell_1/mul_2:z:01lstm_1/while/lstm_cell_1/strided_slice_2:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2#
!lstm_1/while/lstm_cell_1/MatMul_6Ö
lstm_1/while/lstm_cell_1/add_2AddV2+lstm_1/while/lstm_cell_1/BiasAdd_2:output:0+lstm_1/while/lstm_cell_1/MatMul_6:product:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2 
lstm_1/while/lstm_cell_1/add_2
lstm_1/while/lstm_cell_1/TanhTanh"lstm_1/while/lstm_cell_1/add_2:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
lstm_1/while/lstm_cell_1/TanhÃ
lstm_1/while/lstm_cell_1/mul_5Mul$lstm_1/while/lstm_cell_1/Sigmoid:y:0!lstm_1/while/lstm_cell_1/Tanh:y:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2 
lstm_1/while/lstm_cell_1/mul_5Ä
lstm_1/while/lstm_cell_1/add_3AddV2"lstm_1/while/lstm_cell_1/mul_4:z:0"lstm_1/while/lstm_cell_1/mul_5:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2 
lstm_1/while/lstm_cell_1/add_3Ë
)lstm_1/while/lstm_cell_1/ReadVariableOp_3ReadVariableOp2lstm_1_while_lstm_cell_1_readvariableop_resource_0* 
_output_shapes
:
*
dtype02+
)lstm_1/while/lstm_cell_1/ReadVariableOp_3±
.lstm_1/while/lstm_cell_1/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB"       20
.lstm_1/while/lstm_cell_1/strided_slice_3/stackµ
0lstm_1/while/lstm_cell_1/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        22
0lstm_1/while/lstm_cell_1/strided_slice_3/stack_1µ
0lstm_1/while/lstm_cell_1/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      22
0lstm_1/while/lstm_cell_1/strided_slice_3/stack_2 
(lstm_1/while/lstm_cell_1/strided_slice_3StridedSlice1lstm_1/while/lstm_cell_1/ReadVariableOp_3:value:07lstm_1/while/lstm_cell_1/strided_slice_3/stack:output:09lstm_1/while/lstm_cell_1/strided_slice_3/stack_1:output:09lstm_1/while/lstm_cell_1/strided_slice_3/stack_2:output:0*
Index0*
T0* 
_output_shapes
:
*

begin_mask*
end_mask2*
(lstm_1/while/lstm_cell_1/strided_slice_3Ú
!lstm_1/while/lstm_cell_1/MatMul_7MatMul"lstm_1/while/lstm_cell_1/mul_3:z:01lstm_1/while/lstm_cell_1/strided_slice_3:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2#
!lstm_1/while/lstm_cell_1/MatMul_7Ö
lstm_1/while/lstm_cell_1/add_4AddV2+lstm_1/while/lstm_cell_1/BiasAdd_3:output:0+lstm_1/while/lstm_cell_1/MatMul_7:product:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2 
lstm_1/while/lstm_cell_1/add_4ª
"lstm_1/while/lstm_cell_1/Sigmoid_2Sigmoid"lstm_1/while/lstm_cell_1/add_4:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2$
"lstm_1/while/lstm_cell_1/Sigmoid_2¡
lstm_1/while/lstm_cell_1/Tanh_1Tanh"lstm_1/while/lstm_cell_1/add_3:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2!
lstm_1/while/lstm_cell_1/Tanh_1Ç
lstm_1/while/lstm_cell_1/mul_6Mul&lstm_1/while/lstm_cell_1/Sigmoid_2:y:0#lstm_1/while/lstm_cell_1/Tanh_1:y:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2 
lstm_1/while/lstm_cell_1/mul_6
lstm_1/while/Tile/multiplesConst*
_output_shapes
:*
dtype0*
valueB"      2
lstm_1/while/Tile/multiplesÁ
lstm_1/while/TileTile9lstm_1/while/TensorArrayV2Read_1/TensorListGetItem:item:0$lstm_1/while/Tile/multiples:output:0*
T0
*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
lstm_1/while/TileÉ
lstm_1/while/SelectV2SelectV2lstm_1/while/Tile:output:0"lstm_1/while/lstm_cell_1/mul_6:z:0lstm_1_while_placeholder_2*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
lstm_1/while/SelectV2
lstm_1/while/Tile_1/multiplesConst*
_output_shapes
:*
dtype0*
valueB"      2
lstm_1/while/Tile_1/multiplesÇ
lstm_1/while/Tile_1Tile9lstm_1/while/TensorArrayV2Read_1/TensorListGetItem:item:0&lstm_1/while/Tile_1/multiples:output:0*
T0
*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
lstm_1/while/Tile_1
lstm_1/while/Tile_2/multiplesConst*
_output_shapes
:*
dtype0*
valueB"      2
lstm_1/while/Tile_2/multiplesÇ
lstm_1/while/Tile_2Tile9lstm_1/while/TensorArrayV2Read_1/TensorListGetItem:item:0&lstm_1/while/Tile_2/multiples:output:0*
T0
*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
lstm_1/while/Tile_2Ï
lstm_1/while/SelectV2_1SelectV2lstm_1/while/Tile_1:output:0"lstm_1/while/lstm_cell_1/mul_6:z:0lstm_1_while_placeholder_3*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
lstm_1/while/SelectV2_1Ï
lstm_1/while/SelectV2_2SelectV2lstm_1/while/Tile_2:output:0"lstm_1/while/lstm_cell_1/add_3:z:0lstm_1_while_placeholder_4*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
lstm_1/while/SelectV2_2þ
1lstm_1/while/TensorArrayV2Write/TensorListSetItemTensorListSetItemlstm_1_while_placeholder_1lstm_1_while_placeholderlstm_1/while/SelectV2:output:0*
_output_shapes
: *
element_dtype023
1lstm_1/while/TensorArrayV2Write/TensorListSetItemj
lstm_1/while/add/yConst*
_output_shapes
: *
dtype0*
value	B :2
lstm_1/while/add/y
lstm_1/while/addAddV2lstm_1_while_placeholderlstm_1/while/add/y:output:0*
T0*
_output_shapes
: 2
lstm_1/while/addn
lstm_1/while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :2
lstm_1/while/add_1/y
lstm_1/while/add_1AddV2&lstm_1_while_lstm_1_while_loop_counterlstm_1/while/add_1/y:output:0*
T0*
_output_shapes
: 2
lstm_1/while/add_1
lstm_1/while/IdentityIdentitylstm_1/while/add_1:z:0(^lstm_1/while/lstm_cell_1/ReadVariableOp*^lstm_1/while/lstm_cell_1/ReadVariableOp_1*^lstm_1/while/lstm_cell_1/ReadVariableOp_2*^lstm_1/while/lstm_cell_1/ReadVariableOp_3.^lstm_1/while/lstm_cell_1/split/ReadVariableOp0^lstm_1/while/lstm_cell_1/split_1/ReadVariableOp*
T0*
_output_shapes
: 2
lstm_1/while/Identity
lstm_1/while/Identity_1Identity,lstm_1_while_lstm_1_while_maximum_iterations(^lstm_1/while/lstm_cell_1/ReadVariableOp*^lstm_1/while/lstm_cell_1/ReadVariableOp_1*^lstm_1/while/lstm_cell_1/ReadVariableOp_2*^lstm_1/while/lstm_cell_1/ReadVariableOp_3.^lstm_1/while/lstm_cell_1/split/ReadVariableOp0^lstm_1/while/lstm_cell_1/split_1/ReadVariableOp*
T0*
_output_shapes
: 2
lstm_1/while/Identity_1
lstm_1/while/Identity_2Identitylstm_1/while/add:z:0(^lstm_1/while/lstm_cell_1/ReadVariableOp*^lstm_1/while/lstm_cell_1/ReadVariableOp_1*^lstm_1/while/lstm_cell_1/ReadVariableOp_2*^lstm_1/while/lstm_cell_1/ReadVariableOp_3.^lstm_1/while/lstm_cell_1/split/ReadVariableOp0^lstm_1/while/lstm_cell_1/split_1/ReadVariableOp*
T0*
_output_shapes
: 2
lstm_1/while/Identity_2²
lstm_1/while/Identity_3IdentityAlstm_1/while/TensorArrayV2Write/TensorListSetItem:output_handle:0(^lstm_1/while/lstm_cell_1/ReadVariableOp*^lstm_1/while/lstm_cell_1/ReadVariableOp_1*^lstm_1/while/lstm_cell_1/ReadVariableOp_2*^lstm_1/while/lstm_cell_1/ReadVariableOp_3.^lstm_1/while/lstm_cell_1/split/ReadVariableOp0^lstm_1/while/lstm_cell_1/split_1/ReadVariableOp*
T0*
_output_shapes
: 2
lstm_1/while/Identity_3¡
lstm_1/while/Identity_4Identitylstm_1/while/SelectV2:output:0(^lstm_1/while/lstm_cell_1/ReadVariableOp*^lstm_1/while/lstm_cell_1/ReadVariableOp_1*^lstm_1/while/lstm_cell_1/ReadVariableOp_2*^lstm_1/while/lstm_cell_1/ReadVariableOp_3.^lstm_1/while/lstm_cell_1/split/ReadVariableOp0^lstm_1/while/lstm_cell_1/split_1/ReadVariableOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
lstm_1/while/Identity_4£
lstm_1/while/Identity_5Identity lstm_1/while/SelectV2_1:output:0(^lstm_1/while/lstm_cell_1/ReadVariableOp*^lstm_1/while/lstm_cell_1/ReadVariableOp_1*^lstm_1/while/lstm_cell_1/ReadVariableOp_2*^lstm_1/while/lstm_cell_1/ReadVariableOp_3.^lstm_1/while/lstm_cell_1/split/ReadVariableOp0^lstm_1/while/lstm_cell_1/split_1/ReadVariableOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
lstm_1/while/Identity_5£
lstm_1/while/Identity_6Identity lstm_1/while/SelectV2_2:output:0(^lstm_1/while/lstm_cell_1/ReadVariableOp*^lstm_1/while/lstm_cell_1/ReadVariableOp_1*^lstm_1/while/lstm_cell_1/ReadVariableOp_2*^lstm_1/while/lstm_cell_1/ReadVariableOp_3.^lstm_1/while/lstm_cell_1/split/ReadVariableOp0^lstm_1/while/lstm_cell_1/split_1/ReadVariableOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
lstm_1/while/Identity_6"7
lstm_1_while_identitylstm_1/while/Identity:output:0";
lstm_1_while_identity_1 lstm_1/while/Identity_1:output:0";
lstm_1_while_identity_2 lstm_1/while/Identity_2:output:0";
lstm_1_while_identity_3 lstm_1/while/Identity_3:output:0";
lstm_1_while_identity_4 lstm_1/while/Identity_4:output:0";
lstm_1_while_identity_5 lstm_1/while/Identity_5:output:0";
lstm_1_while_identity_6 lstm_1/while/Identity_6:output:0"H
!lstm_1_while_lstm_1_strided_slice#lstm_1_while_lstm_1_strided_slice_0"f
0lstm_1_while_lstm_cell_1_readvariableop_resource2lstm_1_while_lstm_cell_1_readvariableop_resource_0"v
8lstm_1_while_lstm_cell_1_split_1_readvariableop_resource:lstm_1_while_lstm_cell_1_split_1_readvariableop_resource_0"r
6lstm_1_while_lstm_cell_1_split_readvariableop_resource8lstm_1_while_lstm_cell_1_split_readvariableop_resource_0"Ì
clstm_1_while_tensorarrayv2read_1_tensorlistgetitem_lstm_1_tensorarrayunstack_1_tensorlistfromtensorelstm_1_while_tensorarrayv2read_1_tensorlistgetitem_lstm_1_tensorarrayunstack_1_tensorlistfromtensor_0"Ä
_lstm_1_while_tensorarrayv2read_tensorlistgetitem_lstm_1_tensorarrayunstack_tensorlistfromtensoralstm_1_while_tensorarrayv2read_tensorlistgetitem_lstm_1_tensorarrayunstack_tensorlistfromtensor_0*i
_input_shapesX
V: : : : :ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ: : : :::2R
'lstm_1/while/lstm_cell_1/ReadVariableOp'lstm_1/while/lstm_cell_1/ReadVariableOp2V
)lstm_1/while/lstm_cell_1/ReadVariableOp_1)lstm_1/while/lstm_cell_1/ReadVariableOp_12V
)lstm_1/while/lstm_cell_1/ReadVariableOp_2)lstm_1/while/lstm_cell_1/ReadVariableOp_22V
)lstm_1/while/lstm_cell_1/ReadVariableOp_3)lstm_1/while/lstm_cell_1/ReadVariableOp_32^
-lstm_1/while/lstm_cell_1/split/ReadVariableOp-lstm_1/while/lstm_cell_1/split/ReadVariableOp2b
/lstm_1/while/lstm_cell_1/split_1/ReadVariableOp/lstm_1/while/lstm_cell_1/split_1/ReadVariableOp: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :.*
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ:.*
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ:.*
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ:

_output_shapes
: :

_output_shapes
: :	

_output_shapes
: 
üÇ
Õ
A__inference_lstm_1_layer_call_and_return_conditional_losses_16378
inputs_0-
)lstm_cell_1_split_readvariableop_resource/
+lstm_cell_1_split_1_readvariableop_resource'
#lstm_cell_1_readvariableop_resource
identity

identity_1

identity_2¢lstm_cell_1/ReadVariableOp¢lstm_cell_1/ReadVariableOp_1¢lstm_cell_1/ReadVariableOp_2¢lstm_cell_1/ReadVariableOp_3¢ lstm_cell_1/split/ReadVariableOp¢"lstm_cell_1/split_1/ReadVariableOp¢whileF
ShapeShapeinputs_0*
T0*
_output_shapes
:2
Shapet
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice/stackx
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_1x
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_2â
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice]
zeros/mul/yConst*
_output_shapes
: *
dtype0*
value
B :2
zeros/mul/yl
	zeros/mulMulstrided_slice:output:0zeros/mul/y:output:0*
T0*
_output_shapes
: 2
	zeros/mul_
zeros/Less/yConst*
_output_shapes
: *
dtype0*
value
B :è2
zeros/Less/yg

zeros/LessLesszeros/mul:z:0zeros/Less/y:output:0*
T0*
_output_shapes
: 2

zeros/Lessc
zeros/packed/1Const*
_output_shapes
: *
dtype0*
value
B :2
zeros/packed/1
zeros/packedPackstrided_slice:output:0zeros/packed/1:output:0*
N*
T0*
_output_shapes
:2
zeros/packed_
zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
zeros/Constv
zerosFillzeros/packed:output:0zeros/Const:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
zerosa
zeros_1/mul/yConst*
_output_shapes
: *
dtype0*
value
B :2
zeros_1/mul/yr
zeros_1/mulMulstrided_slice:output:0zeros_1/mul/y:output:0*
T0*
_output_shapes
: 2
zeros_1/mulc
zeros_1/Less/yConst*
_output_shapes
: *
dtype0*
value
B :è2
zeros_1/Less/yo
zeros_1/LessLesszeros_1/mul:z:0zeros_1/Less/y:output:0*
T0*
_output_shapes
: 2
zeros_1/Lessg
zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value
B :2
zeros_1/packed/1
zeros_1/packedPackstrided_slice:output:0zeros_1/packed/1:output:0*
N*
T0*
_output_shapes
:2
zeros_1/packedc
zeros_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
zeros_1/Const~
zeros_1Fillzeros_1/packed:output:0zeros_1/Const:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2	
zeros_1u
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
transpose/perm
	transpose	Transposeinputs_0transpose/perm:output:0*
T0*5
_output_shapes#
!:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2
	transposeO
Shape_1Shapetranspose:y:0*
T0*
_output_shapes
:2	
Shape_1x
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_1/stack|
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_1|
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_2î
strided_slice_1StridedSliceShape_1:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice_1
TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ2
TensorArrayV2/element_shape²
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
TensorArrayV2¿
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ   27
5TensorArrayUnstack/TensorListFromTensor/element_shapeø
'TensorArrayUnstack/TensorListFromTensorTensorListFromTensortranspose:y:0>TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type02)
'TensorArrayUnstack/TensorListFromTensorx
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_2/stack|
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_2/stack_1|
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_2/stack_2ý
strided_slice_2StridedSlicetranspose:y:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
shrink_axis_mask2
strided_slice_2x
lstm_cell_1/ones_like/ShapeShapezeros:output:0*
T0*
_output_shapes
:2
lstm_cell_1/ones_like/Shape
lstm_cell_1/ones_like/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ?2
lstm_cell_1/ones_like/Constµ
lstm_cell_1/ones_likeFill$lstm_cell_1/ones_like/Shape:output:0$lstm_cell_1/ones_like/Const:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
lstm_cell_1/ones_like{
lstm_cell_1/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *ä8?2
lstm_cell_1/dropout/Const°
lstm_cell_1/dropout/MulMullstm_cell_1/ones_like:output:0"lstm_cell_1/dropout/Const:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
lstm_cell_1/dropout/Mul
lstm_cell_1/dropout/ShapeShapelstm_cell_1/ones_like:output:0*
T0*
_output_shapes
:2
lstm_cell_1/dropout/Shapeø
0lstm_cell_1/dropout/random_uniform/RandomUniformRandomUniform"lstm_cell_1/dropout/Shape:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
dtype0*
seed±ÿå)*
seed2Éé22
0lstm_cell_1/dropout/random_uniform/RandomUniform
"lstm_cell_1/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *ÍÌÌ=2$
"lstm_cell_1/dropout/GreaterEqual/yï
 lstm_cell_1/dropout/GreaterEqualGreaterEqual9lstm_cell_1/dropout/random_uniform/RandomUniform:output:0+lstm_cell_1/dropout/GreaterEqual/y:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2"
 lstm_cell_1/dropout/GreaterEqual¤
lstm_cell_1/dropout/CastCast$lstm_cell_1/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
lstm_cell_1/dropout/Cast«
lstm_cell_1/dropout/Mul_1Mullstm_cell_1/dropout/Mul:z:0lstm_cell_1/dropout/Cast:y:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
lstm_cell_1/dropout/Mul_1
lstm_cell_1/dropout_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *ä8?2
lstm_cell_1/dropout_1/Const¶
lstm_cell_1/dropout_1/MulMullstm_cell_1/ones_like:output:0$lstm_cell_1/dropout_1/Const:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
lstm_cell_1/dropout_1/Mul
lstm_cell_1/dropout_1/ShapeShapelstm_cell_1/ones_like:output:0*
T0*
_output_shapes
:2
lstm_cell_1/dropout_1/Shapeþ
2lstm_cell_1/dropout_1/random_uniform/RandomUniformRandomUniform$lstm_cell_1/dropout_1/Shape:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
dtype0*
seed±ÿå)*
seed2û°24
2lstm_cell_1/dropout_1/random_uniform/RandomUniform
$lstm_cell_1/dropout_1/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *ÍÌÌ=2&
$lstm_cell_1/dropout_1/GreaterEqual/y÷
"lstm_cell_1/dropout_1/GreaterEqualGreaterEqual;lstm_cell_1/dropout_1/random_uniform/RandomUniform:output:0-lstm_cell_1/dropout_1/GreaterEqual/y:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2$
"lstm_cell_1/dropout_1/GreaterEqualª
lstm_cell_1/dropout_1/CastCast&lstm_cell_1/dropout_1/GreaterEqual:z:0*

DstT0*

SrcT0
*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
lstm_cell_1/dropout_1/Cast³
lstm_cell_1/dropout_1/Mul_1Mullstm_cell_1/dropout_1/Mul:z:0lstm_cell_1/dropout_1/Cast:y:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
lstm_cell_1/dropout_1/Mul_1
lstm_cell_1/dropout_2/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *ä8?2
lstm_cell_1/dropout_2/Const¶
lstm_cell_1/dropout_2/MulMullstm_cell_1/ones_like:output:0$lstm_cell_1/dropout_2/Const:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
lstm_cell_1/dropout_2/Mul
lstm_cell_1/dropout_2/ShapeShapelstm_cell_1/ones_like:output:0*
T0*
_output_shapes
:2
lstm_cell_1/dropout_2/Shapeþ
2lstm_cell_1/dropout_2/random_uniform/RandomUniformRandomUniform$lstm_cell_1/dropout_2/Shape:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
dtype0*
seed±ÿå)*
seed2Ãêº24
2lstm_cell_1/dropout_2/random_uniform/RandomUniform
$lstm_cell_1/dropout_2/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *ÍÌÌ=2&
$lstm_cell_1/dropout_2/GreaterEqual/y÷
"lstm_cell_1/dropout_2/GreaterEqualGreaterEqual;lstm_cell_1/dropout_2/random_uniform/RandomUniform:output:0-lstm_cell_1/dropout_2/GreaterEqual/y:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2$
"lstm_cell_1/dropout_2/GreaterEqualª
lstm_cell_1/dropout_2/CastCast&lstm_cell_1/dropout_2/GreaterEqual:z:0*

DstT0*

SrcT0
*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
lstm_cell_1/dropout_2/Cast³
lstm_cell_1/dropout_2/Mul_1Mullstm_cell_1/dropout_2/Mul:z:0lstm_cell_1/dropout_2/Cast:y:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
lstm_cell_1/dropout_2/Mul_1
lstm_cell_1/dropout_3/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *ä8?2
lstm_cell_1/dropout_3/Const¶
lstm_cell_1/dropout_3/MulMullstm_cell_1/ones_like:output:0$lstm_cell_1/dropout_3/Const:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
lstm_cell_1/dropout_3/Mul
lstm_cell_1/dropout_3/ShapeShapelstm_cell_1/ones_like:output:0*
T0*
_output_shapes
:2
lstm_cell_1/dropout_3/Shapeþ
2lstm_cell_1/dropout_3/random_uniform/RandomUniformRandomUniform$lstm_cell_1/dropout_3/Shape:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
dtype0*
seed±ÿå)*
seed2¹çÕ24
2lstm_cell_1/dropout_3/random_uniform/RandomUniform
$lstm_cell_1/dropout_3/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *ÍÌÌ=2&
$lstm_cell_1/dropout_3/GreaterEqual/y÷
"lstm_cell_1/dropout_3/GreaterEqualGreaterEqual;lstm_cell_1/dropout_3/random_uniform/RandomUniform:output:0-lstm_cell_1/dropout_3/GreaterEqual/y:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2$
"lstm_cell_1/dropout_3/GreaterEqualª
lstm_cell_1/dropout_3/CastCast&lstm_cell_1/dropout_3/GreaterEqual:z:0*

DstT0*

SrcT0
*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
lstm_cell_1/dropout_3/Cast³
lstm_cell_1/dropout_3/Mul_1Mullstm_cell_1/dropout_3/Mul:z:0lstm_cell_1/dropout_3/Cast:y:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
lstm_cell_1/dropout_3/Mul_1h
lstm_cell_1/ConstConst*
_output_shapes
: *
dtype0*
value	B :2
lstm_cell_1/Const|
lstm_cell_1/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2
lstm_cell_1/split/split_dim°
 lstm_cell_1/split/ReadVariableOpReadVariableOp)lstm_cell_1_split_readvariableop_resource* 
_output_shapes
:
*
dtype02"
 lstm_cell_1/split/ReadVariableOpß
lstm_cell_1/splitSplit$lstm_cell_1/split/split_dim:output:0(lstm_cell_1/split/ReadVariableOp:value:0*
T0*D
_output_shapes2
0:
:
:
:
*
	num_split2
lstm_cell_1/split
lstm_cell_1/MatMulMatMulstrided_slice_2:output:0lstm_cell_1/split:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
lstm_cell_1/MatMul
lstm_cell_1/MatMul_1MatMulstrided_slice_2:output:0lstm_cell_1/split:output:1*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
lstm_cell_1/MatMul_1
lstm_cell_1/MatMul_2MatMulstrided_slice_2:output:0lstm_cell_1/split:output:2*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
lstm_cell_1/MatMul_2
lstm_cell_1/MatMul_3MatMulstrided_slice_2:output:0lstm_cell_1/split:output:3*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
lstm_cell_1/MatMul_3l
lstm_cell_1/Const_1Const*
_output_shapes
: *
dtype0*
value	B :2
lstm_cell_1/Const_1
lstm_cell_1/split_1/split_dimConst*
_output_shapes
: *
dtype0*
value	B : 2
lstm_cell_1/split_1/split_dim±
"lstm_cell_1/split_1/ReadVariableOpReadVariableOp+lstm_cell_1_split_1_readvariableop_resource*
_output_shapes	
:*
dtype02$
"lstm_cell_1/split_1/ReadVariableOpÓ
lstm_cell_1/split_1Split&lstm_cell_1/split_1/split_dim:output:0*lstm_cell_1/split_1/ReadVariableOp:value:0*
T0*0
_output_shapes
::::*
	num_split2
lstm_cell_1/split_1¤
lstm_cell_1/BiasAddBiasAddlstm_cell_1/MatMul:product:0lstm_cell_1/split_1:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
lstm_cell_1/BiasAddª
lstm_cell_1/BiasAdd_1BiasAddlstm_cell_1/MatMul_1:product:0lstm_cell_1/split_1:output:1*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
lstm_cell_1/BiasAdd_1ª
lstm_cell_1/BiasAdd_2BiasAddlstm_cell_1/MatMul_2:product:0lstm_cell_1/split_1:output:2*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
lstm_cell_1/BiasAdd_2ª
lstm_cell_1/BiasAdd_3BiasAddlstm_cell_1/MatMul_3:product:0lstm_cell_1/split_1:output:3*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
lstm_cell_1/BiasAdd_3
lstm_cell_1/mulMulzeros:output:0lstm_cell_1/dropout/Mul_1:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
lstm_cell_1/mul
lstm_cell_1/mul_1Mulzeros:output:0lstm_cell_1/dropout_1/Mul_1:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
lstm_cell_1/mul_1
lstm_cell_1/mul_2Mulzeros:output:0lstm_cell_1/dropout_2/Mul_1:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
lstm_cell_1/mul_2
lstm_cell_1/mul_3Mulzeros:output:0lstm_cell_1/dropout_3/Mul_1:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
lstm_cell_1/mul_3
lstm_cell_1/ReadVariableOpReadVariableOp#lstm_cell_1_readvariableop_resource* 
_output_shapes
:
*
dtype02
lstm_cell_1/ReadVariableOp
lstm_cell_1/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB"        2!
lstm_cell_1/strided_slice/stack
!lstm_cell_1/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       2#
!lstm_cell_1/strided_slice/stack_1
!lstm_cell_1/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2#
!lstm_cell_1/strided_slice/stack_2Æ
lstm_cell_1/strided_sliceStridedSlice"lstm_cell_1/ReadVariableOp:value:0(lstm_cell_1/strided_slice/stack:output:0*lstm_cell_1/strided_slice/stack_1:output:0*lstm_cell_1/strided_slice/stack_2:output:0*
Index0*
T0* 
_output_shapes
:
*

begin_mask*
end_mask2
lstm_cell_1/strided_slice¢
lstm_cell_1/MatMul_4MatMullstm_cell_1/mul:z:0"lstm_cell_1/strided_slice:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
lstm_cell_1/MatMul_4
lstm_cell_1/addAddV2lstm_cell_1/BiasAdd:output:0lstm_cell_1/MatMul_4:product:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
lstm_cell_1/add}
lstm_cell_1/SigmoidSigmoidlstm_cell_1/add:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
lstm_cell_1/Sigmoid¢
lstm_cell_1/ReadVariableOp_1ReadVariableOp#lstm_cell_1_readvariableop_resource* 
_output_shapes
:
*
dtype02
lstm_cell_1/ReadVariableOp_1
!lstm_cell_1/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB"       2#
!lstm_cell_1/strided_slice_1/stack
#lstm_cell_1/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       2%
#lstm_cell_1/strided_slice_1/stack_1
#lstm_cell_1/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2%
#lstm_cell_1/strided_slice_1/stack_2Ò
lstm_cell_1/strided_slice_1StridedSlice$lstm_cell_1/ReadVariableOp_1:value:0*lstm_cell_1/strided_slice_1/stack:output:0,lstm_cell_1/strided_slice_1/stack_1:output:0,lstm_cell_1/strided_slice_1/stack_2:output:0*
Index0*
T0* 
_output_shapes
:
*

begin_mask*
end_mask2
lstm_cell_1/strided_slice_1¦
lstm_cell_1/MatMul_5MatMullstm_cell_1/mul_1:z:0$lstm_cell_1/strided_slice_1:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
lstm_cell_1/MatMul_5¢
lstm_cell_1/add_1AddV2lstm_cell_1/BiasAdd_1:output:0lstm_cell_1/MatMul_5:product:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
lstm_cell_1/add_1
lstm_cell_1/Sigmoid_1Sigmoidlstm_cell_1/add_1:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
lstm_cell_1/Sigmoid_1
lstm_cell_1/mul_4Mullstm_cell_1/Sigmoid_1:y:0zeros_1:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
lstm_cell_1/mul_4¢
lstm_cell_1/ReadVariableOp_2ReadVariableOp#lstm_cell_1_readvariableop_resource* 
_output_shapes
:
*
dtype02
lstm_cell_1/ReadVariableOp_2
!lstm_cell_1/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB"       2#
!lstm_cell_1/strided_slice_2/stack
#lstm_cell_1/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       2%
#lstm_cell_1/strided_slice_2/stack_1
#lstm_cell_1/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2%
#lstm_cell_1/strided_slice_2/stack_2Ò
lstm_cell_1/strided_slice_2StridedSlice$lstm_cell_1/ReadVariableOp_2:value:0*lstm_cell_1/strided_slice_2/stack:output:0,lstm_cell_1/strided_slice_2/stack_1:output:0,lstm_cell_1/strided_slice_2/stack_2:output:0*
Index0*
T0* 
_output_shapes
:
*

begin_mask*
end_mask2
lstm_cell_1/strided_slice_2¦
lstm_cell_1/MatMul_6MatMullstm_cell_1/mul_2:z:0$lstm_cell_1/strided_slice_2:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
lstm_cell_1/MatMul_6¢
lstm_cell_1/add_2AddV2lstm_cell_1/BiasAdd_2:output:0lstm_cell_1/MatMul_6:product:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
lstm_cell_1/add_2v
lstm_cell_1/TanhTanhlstm_cell_1/add_2:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
lstm_cell_1/Tanh
lstm_cell_1/mul_5Mullstm_cell_1/Sigmoid:y:0lstm_cell_1/Tanh:y:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
lstm_cell_1/mul_5
lstm_cell_1/add_3AddV2lstm_cell_1/mul_4:z:0lstm_cell_1/mul_5:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
lstm_cell_1/add_3¢
lstm_cell_1/ReadVariableOp_3ReadVariableOp#lstm_cell_1_readvariableop_resource* 
_output_shapes
:
*
dtype02
lstm_cell_1/ReadVariableOp_3
!lstm_cell_1/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB"       2#
!lstm_cell_1/strided_slice_3/stack
#lstm_cell_1/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        2%
#lstm_cell_1/strided_slice_3/stack_1
#lstm_cell_1/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2%
#lstm_cell_1/strided_slice_3/stack_2Ò
lstm_cell_1/strided_slice_3StridedSlice$lstm_cell_1/ReadVariableOp_3:value:0*lstm_cell_1/strided_slice_3/stack:output:0,lstm_cell_1/strided_slice_3/stack_1:output:0,lstm_cell_1/strided_slice_3/stack_2:output:0*
Index0*
T0* 
_output_shapes
:
*

begin_mask*
end_mask2
lstm_cell_1/strided_slice_3¦
lstm_cell_1/MatMul_7MatMullstm_cell_1/mul_3:z:0$lstm_cell_1/strided_slice_3:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
lstm_cell_1/MatMul_7¢
lstm_cell_1/add_4AddV2lstm_cell_1/BiasAdd_3:output:0lstm_cell_1/MatMul_7:product:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
lstm_cell_1/add_4
lstm_cell_1/Sigmoid_2Sigmoidlstm_cell_1/add_4:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
lstm_cell_1/Sigmoid_2z
lstm_cell_1/Tanh_1Tanhlstm_cell_1/add_3:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
lstm_cell_1/Tanh_1
lstm_cell_1/mul_6Mullstm_cell_1/Sigmoid_2:y:0lstm_cell_1/Tanh_1:y:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
lstm_cell_1/mul_6
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ   2
TensorArrayV2_1/element_shape¸
TensorArrayV2_1TensorListReserve&TensorArrayV2_1/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
TensorArrayV2_1N
timeConst*
_output_shapes
: *
dtype0*
value	B : 2
time
while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ2
while/maximum_iterationsj
while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : 2
while/loop_counterã
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0zeros_1:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0)lstm_cell_1_split_readvariableop_resource+lstm_cell_1_split_1_readvariableop_resource#lstm_cell_1_readvariableop_resource*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*N
_output_shapes<
:: : : : :ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ: : : : : *%
_read_only_resource_inputs
	
*
bodyR
while_body_16215*
condR
while_cond_16214*M
output_shapes<
:: : : : :ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ: : : : : *
parallel_iterations 2
whileµ
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ   22
0TensorArrayV2Stack/TensorListStack/element_shapeò
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*5
_output_shapes#
!:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ*
element_dtype02$
"TensorArrayV2Stack/TensorListStack
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
ÿÿÿÿÿÿÿÿÿ2
strided_slice_3/stack|
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_3/stack_1|
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_3/stack_2
strided_slice_3StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
shrink_axis_mask2
strided_slice_3y
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
transpose_1/perm¯
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*5
_output_shapes#
!:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2
transpose_1f
runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    2	
runtime»
IdentityIdentitytranspose_1:y:0^lstm_cell_1/ReadVariableOp^lstm_cell_1/ReadVariableOp_1^lstm_cell_1/ReadVariableOp_2^lstm_cell_1/ReadVariableOp_3!^lstm_cell_1/split/ReadVariableOp#^lstm_cell_1/split_1/ReadVariableOp^while*
T0*5
_output_shapes#
!:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2

Identity±

Identity_1Identitywhile:output:4^lstm_cell_1/ReadVariableOp^lstm_cell_1/ReadVariableOp_1^lstm_cell_1/ReadVariableOp_2^lstm_cell_1/ReadVariableOp_3!^lstm_cell_1/split/ReadVariableOp#^lstm_cell_1/split_1/ReadVariableOp^while*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity_1±

Identity_2Identitywhile:output:5^lstm_cell_1/ReadVariableOp^lstm_cell_1/ReadVariableOp_1^lstm_cell_1/ReadVariableOp_2^lstm_cell_1/ReadVariableOp_3!^lstm_cell_1/split/ReadVariableOp#^lstm_cell_1/split_1/ReadVariableOp^while*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity_2"
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0*@
_input_shapes/
-:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ:::28
lstm_cell_1/ReadVariableOplstm_cell_1/ReadVariableOp2<
lstm_cell_1/ReadVariableOp_1lstm_cell_1/ReadVariableOp_12<
lstm_cell_1/ReadVariableOp_2lstm_cell_1/ReadVariableOp_22<
lstm_cell_1/ReadVariableOp_3lstm_cell_1/ReadVariableOp_32D
 lstm_cell_1/split/ReadVariableOp lstm_cell_1/split/ReadVariableOp2H
"lstm_cell_1/split_1/ReadVariableOp"lstm_cell_1/split_1/ReadVariableOp2
whilewhile:_ [
5
_output_shapes#
!:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
"
_user_specified_name
inputs/0

Þ
&__inference_lstm_1_layer_call_fn_17237

inputs
mask

initial_state_0
initial_state_1
unknown
	unknown_0
	unknown_1
identity

identity_1

identity_2¢StatefulPartitionedCallá
StatefulPartitionedCallStatefulPartitionedCallinputsmaskinitial_state_0initial_state_1unknown	unknown_0	unknown_1*
Tin
	2
*
Tout
2*
_collective_manager_ids
 *]
_output_shapesK
I:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *J
fERC
A__inference_lstm_1_layer_call_and_return_conditional_losses_147992
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*5
_output_shapes#
!:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2

Identity

Identity_1Identity StatefulPartitionedCall:output:1^StatefulPartitionedCall*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity_1

Identity_2Identity StatefulPartitionedCall:output:2^StatefulPartitionedCall*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity_2"
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0*
_input_shapess
q:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:::22
StatefulPartitionedCallStatefulPartitionedCall:] Y
5
_output_shapes#
!:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs:VR
0
_output_shapes
:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ

_user_specified_namemask:YU
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
)
_user_specified_nameinitial_state/0:YU
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
)
_user_specified_nameinitial_state/1
ìG
ó
A__inference_lstm_1_layer_call_and_return_conditional_losses_14260

inputs
lstm_cell_1_14176
lstm_cell_1_14178
lstm_cell_1_14180
identity

identity_1

identity_2¢#lstm_cell_1/StatefulPartitionedCall¢whileD
ShapeShapeinputs*
T0*
_output_shapes
:2
Shapet
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice/stackx
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_1x
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_2â
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice]
zeros/mul/yConst*
_output_shapes
: *
dtype0*
value
B :2
zeros/mul/yl
	zeros/mulMulstrided_slice:output:0zeros/mul/y:output:0*
T0*
_output_shapes
: 2
	zeros/mul_
zeros/Less/yConst*
_output_shapes
: *
dtype0*
value
B :è2
zeros/Less/yg

zeros/LessLesszeros/mul:z:0zeros/Less/y:output:0*
T0*
_output_shapes
: 2

zeros/Lessc
zeros/packed/1Const*
_output_shapes
: *
dtype0*
value
B :2
zeros/packed/1
zeros/packedPackstrided_slice:output:0zeros/packed/1:output:0*
N*
T0*
_output_shapes
:2
zeros/packed_
zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
zeros/Constv
zerosFillzeros/packed:output:0zeros/Const:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
zerosa
zeros_1/mul/yConst*
_output_shapes
: *
dtype0*
value
B :2
zeros_1/mul/yr
zeros_1/mulMulstrided_slice:output:0zeros_1/mul/y:output:0*
T0*
_output_shapes
: 2
zeros_1/mulc
zeros_1/Less/yConst*
_output_shapes
: *
dtype0*
value
B :è2
zeros_1/Less/yo
zeros_1/LessLesszeros_1/mul:z:0zeros_1/Less/y:output:0*
T0*
_output_shapes
: 2
zeros_1/Lessg
zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value
B :2
zeros_1/packed/1
zeros_1/packedPackstrided_slice:output:0zeros_1/packed/1:output:0*
N*
T0*
_output_shapes
:2
zeros_1/packedc
zeros_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
zeros_1/Const~
zeros_1Fillzeros_1/packed:output:0zeros_1/Const:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2	
zeros_1u
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
transpose/perm
	transpose	Transposeinputstranspose/perm:output:0*
T0*5
_output_shapes#
!:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2
	transposeO
Shape_1Shapetranspose:y:0*
T0*
_output_shapes
:2	
Shape_1x
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_1/stack|
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_1|
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_2î
strided_slice_1StridedSliceShape_1:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice_1
TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ2
TensorArrayV2/element_shape²
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
TensorArrayV2¿
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ   27
5TensorArrayUnstack/TensorListFromTensor/element_shapeø
'TensorArrayUnstack/TensorListFromTensorTensorListFromTensortranspose:y:0>TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type02)
'TensorArrayUnstack/TensorListFromTensorx
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_2/stack|
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_2/stack_1|
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_2/stack_2ý
strided_slice_2StridedSlicetranspose:y:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
shrink_axis_mask2
strided_slice_2
#lstm_cell_1/StatefulPartitionedCallStatefulPartitionedCallstrided_slice_2:output:0zeros:output:0zeros_1:output:0lstm_cell_1_14176lstm_cell_1_14178lstm_cell_1_14180*
Tin

2*
Tout
2*
_collective_manager_ids
 *P
_output_shapes>
<:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *O
fJRH
F__inference_lstm_cell_1_layer_call_and_return_conditional_losses_138142%
#lstm_cell_1/StatefulPartitionedCall
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ   2
TensorArrayV2_1/element_shape¸
TensorArrayV2_1TensorListReserve&TensorArrayV2_1/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
TensorArrayV2_1N
timeConst*
_output_shapes
: *
dtype0*
value	B : 2
time
while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ2
while/maximum_iterationsj
while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : 2
while/loop_counter
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0zeros_1:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0lstm_cell_1_14176lstm_cell_1_14178lstm_cell_1_14180*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*N
_output_shapes<
:: : : : :ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ: : : : : *%
_read_only_resource_inputs
	
*
bodyR
while_body_14189*
condR
while_cond_14188*M
output_shapes<
:: : : : :ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ: : : : : *
parallel_iterations 2
whileµ
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ   22
0TensorArrayV2Stack/TensorListStack/element_shapeò
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*5
_output_shapes#
!:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ*
element_dtype02$
"TensorArrayV2Stack/TensorListStack
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
ÿÿÿÿÿÿÿÿÿ2
strided_slice_3/stack|
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_3/stack_1|
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_3/stack_2
strided_slice_3StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
shrink_axis_mask2
strided_slice_3y
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
transpose_1/perm¯
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*5
_output_shapes#
!:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2
transpose_1f
runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    2	
runtime
IdentityIdentitytranspose_1:y:0$^lstm_cell_1/StatefulPartitionedCall^while*
T0*5
_output_shapes#
!:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2

Identity

Identity_1Identitywhile:output:4$^lstm_cell_1/StatefulPartitionedCall^while*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity_1

Identity_2Identitywhile:output:5$^lstm_cell_1/StatefulPartitionedCall^while*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity_2"
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0*@
_input_shapes/
-:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ:::2J
#lstm_cell_1/StatefulPartitionedCall#lstm_cell_1/StatefulPartitionedCall2
whilewhile:] Y
5
_output_shapes#
!:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
ðl

F__inference_lstm_cell_1_layer_call_and_return_conditional_losses_17410

inputs
states_0
states_1!
split_readvariableop_resource#
split_1_readvariableop_resource
readvariableop_resource
identity

identity_1

identity_2¢ReadVariableOp¢ReadVariableOp_1¢ReadVariableOp_2¢ReadVariableOp_3¢split/ReadVariableOp¢split_1/ReadVariableOpZ
ones_like/ShapeShapestates_0*
T0*
_output_shapes
:2
ones_like/Shapeg
ones_like/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ?2
ones_like/Const
	ones_likeFillones_like/Shape:output:0ones_like/Const:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
	ones_likec
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *ä8?2
dropout/Const
dropout/MulMulones_like:output:0dropout/Const:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
dropout/Mul`
dropout/ShapeShapeones_like:output:0*
T0*
_output_shapes
:2
dropout/ShapeÔ
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
dtype0*
seed±ÿå)*
seed2âãÄ2&
$dropout/random_uniform/RandomUniformu
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *ÍÌÌ=2
dropout/GreaterEqual/y¿
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
dropout/GreaterEqual
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
dropout/Cast{
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
dropout/Mul_1g
dropout_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *ä8?2
dropout_1/Const
dropout_1/MulMulones_like:output:0dropout_1/Const:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
dropout_1/Muld
dropout_1/ShapeShapeones_like:output:0*
T0*
_output_shapes
:2
dropout_1/ShapeÚ
&dropout_1/random_uniform/RandomUniformRandomUniformdropout_1/Shape:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
dtype0*
seed±ÿå)*
seed2Ëó¥2(
&dropout_1/random_uniform/RandomUniformy
dropout_1/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *ÍÌÌ=2
dropout_1/GreaterEqual/yÇ
dropout_1/GreaterEqualGreaterEqual/dropout_1/random_uniform/RandomUniform:output:0!dropout_1/GreaterEqual/y:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
dropout_1/GreaterEqual
dropout_1/CastCastdropout_1/GreaterEqual:z:0*

DstT0*

SrcT0
*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
dropout_1/Cast
dropout_1/Mul_1Muldropout_1/Mul:z:0dropout_1/Cast:y:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
dropout_1/Mul_1g
dropout_2/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *ä8?2
dropout_2/Const
dropout_2/MulMulones_like:output:0dropout_2/Const:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
dropout_2/Muld
dropout_2/ShapeShapeones_like:output:0*
T0*
_output_shapes
:2
dropout_2/ShapeÙ
&dropout_2/random_uniform/RandomUniformRandomUniformdropout_2/Shape:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
dtype0*
seed±ÿå)*
seed2ÜO2(
&dropout_2/random_uniform/RandomUniformy
dropout_2/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *ÍÌÌ=2
dropout_2/GreaterEqual/yÇ
dropout_2/GreaterEqualGreaterEqual/dropout_2/random_uniform/RandomUniform:output:0!dropout_2/GreaterEqual/y:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
dropout_2/GreaterEqual
dropout_2/CastCastdropout_2/GreaterEqual:z:0*

DstT0*

SrcT0
*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
dropout_2/Cast
dropout_2/Mul_1Muldropout_2/Mul:z:0dropout_2/Cast:y:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
dropout_2/Mul_1g
dropout_3/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *ä8?2
dropout_3/Const
dropout_3/MulMulones_like:output:0dropout_3/Const:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
dropout_3/Muld
dropout_3/ShapeShapeones_like:output:0*
T0*
_output_shapes
:2
dropout_3/ShapeÚ
&dropout_3/random_uniform/RandomUniformRandomUniformdropout_3/Shape:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
dtype0*
seed±ÿå)*
seed2µÁ2(
&dropout_3/random_uniform/RandomUniformy
dropout_3/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *ÍÌÌ=2
dropout_3/GreaterEqual/yÇ
dropout_3/GreaterEqualGreaterEqual/dropout_3/random_uniform/RandomUniform:output:0!dropout_3/GreaterEqual/y:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
dropout_3/GreaterEqual
dropout_3/CastCastdropout_3/GreaterEqual:z:0*

DstT0*

SrcT0
*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
dropout_3/Cast
dropout_3/Mul_1Muldropout_3/Mul:z:0dropout_3/Cast:y:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
dropout_3/Mul_1P
ConstConst*
_output_shapes
: *
dtype0*
value	B :2
Constd
split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2
split/split_dim
split/ReadVariableOpReadVariableOpsplit_readvariableop_resource* 
_output_shapes
:
*
dtype02
split/ReadVariableOp¯
splitSplitsplit/split_dim:output:0split/ReadVariableOp:value:0*
T0*D
_output_shapes2
0:
:
:
:
*
	num_split2
splite
MatMulMatMulinputssplit:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
MatMuli
MatMul_1MatMulinputssplit:output:1*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

MatMul_1i
MatMul_2MatMulinputssplit:output:2*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

MatMul_2i
MatMul_3MatMulinputssplit:output:3*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

MatMul_3T
Const_1Const*
_output_shapes
: *
dtype0*
value	B :2	
Const_1h
split_1/split_dimConst*
_output_shapes
: *
dtype0*
value	B : 2
split_1/split_dim
split_1/ReadVariableOpReadVariableOpsplit_1_readvariableop_resource*
_output_shapes	
:*
dtype02
split_1/ReadVariableOp£
split_1Splitsplit_1/split_dim:output:0split_1/ReadVariableOp:value:0*
T0*0
_output_shapes
::::*
	num_split2	
split_1t
BiasAddBiasAddMatMul:product:0split_1:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2	
BiasAddz
	BiasAdd_1BiasAddMatMul_1:product:0split_1:output:1*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
	BiasAdd_1z
	BiasAdd_2BiasAddMatMul_2:product:0split_1:output:2*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
	BiasAdd_2z
	BiasAdd_3BiasAddMatMul_3:product:0split_1:output:3*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
	BiasAdd_3a
mulMulstates_0dropout/Mul_1:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
mulg
mul_1Mulstates_0dropout_1/Mul_1:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
mul_1g
mul_2Mulstates_0dropout_2/Mul_1:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
mul_2g
mul_3Mulstates_0dropout_3/Mul_1:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
mul_3z
ReadVariableOpReadVariableOpreadvariableop_resource* 
_output_shapes
:
*
dtype02
ReadVariableOp{
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB"        2
strided_slice/stack
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       2
strided_slice/stack_1
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2
strided_slice/stack_2þ
strided_sliceStridedSliceReadVariableOp:value:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0* 
_output_shapes
:
*

begin_mask*
end_mask2
strided_slicer
MatMul_4MatMulmul:z:0strided_slice:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

MatMul_4l
addAddV2BiasAdd:output:0MatMul_4:product:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
addY
SigmoidSigmoidadd:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2	
Sigmoid~
ReadVariableOp_1ReadVariableOpreadvariableop_resource* 
_output_shapes
:
*
dtype02
ReadVariableOp_1
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB"       2
strided_slice_1/stack
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       2
strided_slice_1/stack_1
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2
strided_slice_1/stack_2
strided_slice_1StridedSliceReadVariableOp_1:value:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0* 
_output_shapes
:
*

begin_mask*
end_mask2
strided_slice_1v
MatMul_5MatMul	mul_1:z:0strided_slice_1:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

MatMul_5r
add_1AddV2BiasAdd_1:output:0MatMul_5:product:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
add_1_
	Sigmoid_1Sigmoid	add_1:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
	Sigmoid_1a
mul_4MulSigmoid_1:y:0states_1*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
mul_4~
ReadVariableOp_2ReadVariableOpreadvariableop_resource* 
_output_shapes
:
*
dtype02
ReadVariableOp_2
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB"       2
strided_slice_2/stack
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       2
strided_slice_2/stack_1
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2
strided_slice_2/stack_2
strided_slice_2StridedSliceReadVariableOp_2:value:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0* 
_output_shapes
:
*

begin_mask*
end_mask2
strided_slice_2v
MatMul_6MatMul	mul_2:z:0strided_slice_2:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

MatMul_6r
add_2AddV2BiasAdd_2:output:0MatMul_6:product:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
add_2R
TanhTanh	add_2:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
Tanh_
mul_5MulSigmoid:y:0Tanh:y:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
mul_5`
add_3AddV2	mul_4:z:0	mul_5:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
add_3~
ReadVariableOp_3ReadVariableOpreadvariableop_resource* 
_output_shapes
:
*
dtype02
ReadVariableOp_3
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB"       2
strided_slice_3/stack
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        2
strided_slice_3/stack_1
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2
strided_slice_3/stack_2
strided_slice_3StridedSliceReadVariableOp_3:value:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0* 
_output_shapes
:
*

begin_mask*
end_mask2
strided_slice_3v
MatMul_7MatMul	mul_3:z:0strided_slice_3:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

MatMul_7r
add_4AddV2BiasAdd_3:output:0MatMul_7:product:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
add_4_
	Sigmoid_2Sigmoid	add_4:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
	Sigmoid_2V
Tanh_1Tanh	add_3:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
Tanh_1c
mul_6MulSigmoid_2:y:0
Tanh_1:y:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
mul_6Ø
IdentityIdentity	mul_6:z:0^ReadVariableOp^ReadVariableOp_1^ReadVariableOp_2^ReadVariableOp_3^split/ReadVariableOp^split_1/ReadVariableOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

IdentityÜ

Identity_1Identity	mul_6:z:0^ReadVariableOp^ReadVariableOp_1^ReadVariableOp_2^ReadVariableOp_3^split/ReadVariableOp^split_1/ReadVariableOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity_1Ü

Identity_2Identity	add_3:z:0^ReadVariableOp^ReadVariableOp_1^ReadVariableOp_2^ReadVariableOp_3^split/ReadVariableOp^split_1/ReadVariableOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity_2"
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0*[
_input_shapesJ
H:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:::2 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_12$
ReadVariableOp_2ReadVariableOp_22$
ReadVariableOp_3ReadVariableOp_32,
split/ReadVariableOpsplit/ReadVariableOp20
split_1/ReadVariableOpsplit_1/ReadVariableOp:P L
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs:RN
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
"
_user_specified_name
states/0:RN
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
"
_user_specified_name
states/1
¶

A__inference_lstm_1_layer_call_and_return_conditional_losses_17219

inputs
mask

initial_state_0
initial_state_1-
)lstm_cell_1_split_readvariableop_resource/
+lstm_cell_1_split_1_readvariableop_resource'
#lstm_cell_1_readvariableop_resource
identity

identity_1

identity_2¢lstm_cell_1/ReadVariableOp¢lstm_cell_1/ReadVariableOp_1¢lstm_cell_1/ReadVariableOp_2¢lstm_cell_1/ReadVariableOp_3¢ lstm_cell_1/split/ReadVariableOp¢"lstm_cell_1/split_1/ReadVariableOp¢whileu
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
transpose/perm
	transpose	Transposeinputstranspose/perm:output:0*
T0*5
_output_shapes#
!:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2
	transposeK
ShapeShapetranspose:y:0*
T0*
_output_shapes
:2
Shapet
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice/stackx
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_1x
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_2â
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slicek
ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ2
ExpandDims/dim

ExpandDims
ExpandDimsmaskExpandDims/dim:output:0*
T0
*4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2

ExpandDimsy
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
transpose_1/perm
transpose_1	TransposeExpandDims:output:0transpose_1/perm:output:0*
T0
*4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2
transpose_1
TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ2
TensorArrayV2/element_shape°
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
TensorArrayV2¿
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ   27
5TensorArrayUnstack/TensorListFromTensor/element_shapeø
'TensorArrayUnstack/TensorListFromTensorTensorListFromTensortranspose:y:0>TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type02)
'TensorArrayUnstack/TensorListFromTensorx
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_1/stack|
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_1|
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_2ý
strided_slice_1StridedSlicetranspose:y:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
shrink_axis_mask2
strided_slice_1y
lstm_cell_1/ones_like/ShapeShapeinitial_state_0*
T0*
_output_shapes
:2
lstm_cell_1/ones_like/Shape
lstm_cell_1/ones_like/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ?2
lstm_cell_1/ones_like/Constµ
lstm_cell_1/ones_likeFill$lstm_cell_1/ones_like/Shape:output:0$lstm_cell_1/ones_like/Const:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
lstm_cell_1/ones_likeh
lstm_cell_1/ConstConst*
_output_shapes
: *
dtype0*
value	B :2
lstm_cell_1/Const|
lstm_cell_1/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2
lstm_cell_1/split/split_dim°
 lstm_cell_1/split/ReadVariableOpReadVariableOp)lstm_cell_1_split_readvariableop_resource* 
_output_shapes
:
*
dtype02"
 lstm_cell_1/split/ReadVariableOpß
lstm_cell_1/splitSplit$lstm_cell_1/split/split_dim:output:0(lstm_cell_1/split/ReadVariableOp:value:0*
T0*D
_output_shapes2
0:
:
:
:
*
	num_split2
lstm_cell_1/split
lstm_cell_1/MatMulMatMulstrided_slice_1:output:0lstm_cell_1/split:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
lstm_cell_1/MatMul
lstm_cell_1/MatMul_1MatMulstrided_slice_1:output:0lstm_cell_1/split:output:1*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
lstm_cell_1/MatMul_1
lstm_cell_1/MatMul_2MatMulstrided_slice_1:output:0lstm_cell_1/split:output:2*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
lstm_cell_1/MatMul_2
lstm_cell_1/MatMul_3MatMulstrided_slice_1:output:0lstm_cell_1/split:output:3*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
lstm_cell_1/MatMul_3l
lstm_cell_1/Const_1Const*
_output_shapes
: *
dtype0*
value	B :2
lstm_cell_1/Const_1
lstm_cell_1/split_1/split_dimConst*
_output_shapes
: *
dtype0*
value	B : 2
lstm_cell_1/split_1/split_dim±
"lstm_cell_1/split_1/ReadVariableOpReadVariableOp+lstm_cell_1_split_1_readvariableop_resource*
_output_shapes	
:*
dtype02$
"lstm_cell_1/split_1/ReadVariableOpÓ
lstm_cell_1/split_1Split&lstm_cell_1/split_1/split_dim:output:0*lstm_cell_1/split_1/ReadVariableOp:value:0*
T0*0
_output_shapes
::::*
	num_split2
lstm_cell_1/split_1¤
lstm_cell_1/BiasAddBiasAddlstm_cell_1/MatMul:product:0lstm_cell_1/split_1:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
lstm_cell_1/BiasAddª
lstm_cell_1/BiasAdd_1BiasAddlstm_cell_1/MatMul_1:product:0lstm_cell_1/split_1:output:1*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
lstm_cell_1/BiasAdd_1ª
lstm_cell_1/BiasAdd_2BiasAddlstm_cell_1/MatMul_2:product:0lstm_cell_1/split_1:output:2*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
lstm_cell_1/BiasAdd_2ª
lstm_cell_1/BiasAdd_3BiasAddlstm_cell_1/MatMul_3:product:0lstm_cell_1/split_1:output:3*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
lstm_cell_1/BiasAdd_3
lstm_cell_1/mulMulinitial_state_0lstm_cell_1/ones_like:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
lstm_cell_1/mul
lstm_cell_1/mul_1Mulinitial_state_0lstm_cell_1/ones_like:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
lstm_cell_1/mul_1
lstm_cell_1/mul_2Mulinitial_state_0lstm_cell_1/ones_like:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
lstm_cell_1/mul_2
lstm_cell_1/mul_3Mulinitial_state_0lstm_cell_1/ones_like:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
lstm_cell_1/mul_3
lstm_cell_1/ReadVariableOpReadVariableOp#lstm_cell_1_readvariableop_resource* 
_output_shapes
:
*
dtype02
lstm_cell_1/ReadVariableOp
lstm_cell_1/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB"        2!
lstm_cell_1/strided_slice/stack
!lstm_cell_1/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       2#
!lstm_cell_1/strided_slice/stack_1
!lstm_cell_1/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2#
!lstm_cell_1/strided_slice/stack_2Æ
lstm_cell_1/strided_sliceStridedSlice"lstm_cell_1/ReadVariableOp:value:0(lstm_cell_1/strided_slice/stack:output:0*lstm_cell_1/strided_slice/stack_1:output:0*lstm_cell_1/strided_slice/stack_2:output:0*
Index0*
T0* 
_output_shapes
:
*

begin_mask*
end_mask2
lstm_cell_1/strided_slice¢
lstm_cell_1/MatMul_4MatMullstm_cell_1/mul:z:0"lstm_cell_1/strided_slice:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
lstm_cell_1/MatMul_4
lstm_cell_1/addAddV2lstm_cell_1/BiasAdd:output:0lstm_cell_1/MatMul_4:product:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
lstm_cell_1/add}
lstm_cell_1/SigmoidSigmoidlstm_cell_1/add:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
lstm_cell_1/Sigmoid¢
lstm_cell_1/ReadVariableOp_1ReadVariableOp#lstm_cell_1_readvariableop_resource* 
_output_shapes
:
*
dtype02
lstm_cell_1/ReadVariableOp_1
!lstm_cell_1/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB"       2#
!lstm_cell_1/strided_slice_1/stack
#lstm_cell_1/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       2%
#lstm_cell_1/strided_slice_1/stack_1
#lstm_cell_1/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2%
#lstm_cell_1/strided_slice_1/stack_2Ò
lstm_cell_1/strided_slice_1StridedSlice$lstm_cell_1/ReadVariableOp_1:value:0*lstm_cell_1/strided_slice_1/stack:output:0,lstm_cell_1/strided_slice_1/stack_1:output:0,lstm_cell_1/strided_slice_1/stack_2:output:0*
Index0*
T0* 
_output_shapes
:
*

begin_mask*
end_mask2
lstm_cell_1/strided_slice_1¦
lstm_cell_1/MatMul_5MatMullstm_cell_1/mul_1:z:0$lstm_cell_1/strided_slice_1:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
lstm_cell_1/MatMul_5¢
lstm_cell_1/add_1AddV2lstm_cell_1/BiasAdd_1:output:0lstm_cell_1/MatMul_5:product:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
lstm_cell_1/add_1
lstm_cell_1/Sigmoid_1Sigmoidlstm_cell_1/add_1:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
lstm_cell_1/Sigmoid_1
lstm_cell_1/mul_4Mullstm_cell_1/Sigmoid_1:y:0initial_state_1*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
lstm_cell_1/mul_4¢
lstm_cell_1/ReadVariableOp_2ReadVariableOp#lstm_cell_1_readvariableop_resource* 
_output_shapes
:
*
dtype02
lstm_cell_1/ReadVariableOp_2
!lstm_cell_1/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB"       2#
!lstm_cell_1/strided_slice_2/stack
#lstm_cell_1/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       2%
#lstm_cell_1/strided_slice_2/stack_1
#lstm_cell_1/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2%
#lstm_cell_1/strided_slice_2/stack_2Ò
lstm_cell_1/strided_slice_2StridedSlice$lstm_cell_1/ReadVariableOp_2:value:0*lstm_cell_1/strided_slice_2/stack:output:0,lstm_cell_1/strided_slice_2/stack_1:output:0,lstm_cell_1/strided_slice_2/stack_2:output:0*
Index0*
T0* 
_output_shapes
:
*

begin_mask*
end_mask2
lstm_cell_1/strided_slice_2¦
lstm_cell_1/MatMul_6MatMullstm_cell_1/mul_2:z:0$lstm_cell_1/strided_slice_2:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
lstm_cell_1/MatMul_6¢
lstm_cell_1/add_2AddV2lstm_cell_1/BiasAdd_2:output:0lstm_cell_1/MatMul_6:product:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
lstm_cell_1/add_2v
lstm_cell_1/TanhTanhlstm_cell_1/add_2:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
lstm_cell_1/Tanh
lstm_cell_1/mul_5Mullstm_cell_1/Sigmoid:y:0lstm_cell_1/Tanh:y:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
lstm_cell_1/mul_5
lstm_cell_1/add_3AddV2lstm_cell_1/mul_4:z:0lstm_cell_1/mul_5:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
lstm_cell_1/add_3¢
lstm_cell_1/ReadVariableOp_3ReadVariableOp#lstm_cell_1_readvariableop_resource* 
_output_shapes
:
*
dtype02
lstm_cell_1/ReadVariableOp_3
!lstm_cell_1/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB"       2#
!lstm_cell_1/strided_slice_3/stack
#lstm_cell_1/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        2%
#lstm_cell_1/strided_slice_3/stack_1
#lstm_cell_1/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2%
#lstm_cell_1/strided_slice_3/stack_2Ò
lstm_cell_1/strided_slice_3StridedSlice$lstm_cell_1/ReadVariableOp_3:value:0*lstm_cell_1/strided_slice_3/stack:output:0,lstm_cell_1/strided_slice_3/stack_1:output:0,lstm_cell_1/strided_slice_3/stack_2:output:0*
Index0*
T0* 
_output_shapes
:
*

begin_mask*
end_mask2
lstm_cell_1/strided_slice_3¦
lstm_cell_1/MatMul_7MatMullstm_cell_1/mul_3:z:0$lstm_cell_1/strided_slice_3:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
lstm_cell_1/MatMul_7¢
lstm_cell_1/add_4AddV2lstm_cell_1/BiasAdd_3:output:0lstm_cell_1/MatMul_7:product:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
lstm_cell_1/add_4
lstm_cell_1/Sigmoid_2Sigmoidlstm_cell_1/add_4:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
lstm_cell_1/Sigmoid_2z
lstm_cell_1/Tanh_1Tanhlstm_cell_1/add_3:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
lstm_cell_1/Tanh_1
lstm_cell_1/mul_6Mullstm_cell_1/Sigmoid_2:y:0lstm_cell_1/Tanh_1:y:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
lstm_cell_1/mul_6
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ   2
TensorArrayV2_1/element_shape¶
TensorArrayV2_1TensorListReserve&TensorArrayV2_1/element_shape:output:0strided_slice:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
TensorArrayV2_1N
timeConst*
_output_shapes
: *
dtype0*
value	B : 2
time
TensorArrayV2_2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ2
TensorArrayV2_2/element_shape¶
TensorArrayV2_2TensorListReserve&TensorArrayV2_2/element_shape:output:0strided_slice:output:0*
_output_shapes
: *
element_dtype0
*

shape_type02
TensorArrayV2_2Ã
7TensorArrayUnstack_1/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ   29
7TensorArrayUnstack_1/TensorListFromTensor/element_shape
)TensorArrayUnstack_1/TensorListFromTensorTensorListFromTensortranspose_1:y:0@TensorArrayUnstack_1/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0
*

shape_type02+
)TensorArrayUnstack_1/TensorListFromTensoro

zeros_like	ZerosLikelstm_cell_1/mul_6:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

zeros_like
while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ2
while/maximum_iterationsj
while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : 2
while/loop_counterÚ
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros_like:y:0initial_state_0initial_state_1strided_slice:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:09TensorArrayUnstack_1/TensorListFromTensor:output_handle:0)lstm_cell_1_split_readvariableop_resource+lstm_cell_1_split_1_readvariableop_resource#lstm_cell_1_readvariableop_resource*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*d
_output_shapesR
P: : : : :ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ: : : : : : *%
_read_only_resource_inputs

*
bodyR
while_body_17070*
condR
while_cond_17069*c
output_shapesR
P: : : : :ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ: : : : : : *
parallel_iterations 2
whileµ
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ   22
0TensorArrayV2Stack/TensorListStack/element_shapeò
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*5
_output_shapes#
!:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ*
element_dtype02$
"TensorArrayV2Stack/TensorListStack
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB:
ÿÿÿÿÿÿÿÿÿ2
strided_slice_2/stack|
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_2/stack_1|
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_2/stack_2
strided_slice_2StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
shrink_axis_mask2
strided_slice_2y
transpose_2/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
transpose_2/perm¯
transpose_2	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_2/perm:output:0*
T0*5
_output_shapes#
!:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2
transpose_2f
runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    2	
runtime»
IdentityIdentitytranspose_2:y:0^lstm_cell_1/ReadVariableOp^lstm_cell_1/ReadVariableOp_1^lstm_cell_1/ReadVariableOp_2^lstm_cell_1/ReadVariableOp_3!^lstm_cell_1/split/ReadVariableOp#^lstm_cell_1/split_1/ReadVariableOp^while*
T0*5
_output_shapes#
!:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2

Identity±

Identity_1Identitywhile:output:5^lstm_cell_1/ReadVariableOp^lstm_cell_1/ReadVariableOp_1^lstm_cell_1/ReadVariableOp_2^lstm_cell_1/ReadVariableOp_3!^lstm_cell_1/split/ReadVariableOp#^lstm_cell_1/split_1/ReadVariableOp^while*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity_1±

Identity_2Identitywhile:output:6^lstm_cell_1/ReadVariableOp^lstm_cell_1/ReadVariableOp_1^lstm_cell_1/ReadVariableOp_2^lstm_cell_1/ReadVariableOp_3!^lstm_cell_1/split/ReadVariableOp#^lstm_cell_1/split_1/ReadVariableOp^while*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity_2"
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0*
_input_shapess
q:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:::28
lstm_cell_1/ReadVariableOplstm_cell_1/ReadVariableOp2<
lstm_cell_1/ReadVariableOp_1lstm_cell_1/ReadVariableOp_12<
lstm_cell_1/ReadVariableOp_2lstm_cell_1/ReadVariableOp_22<
lstm_cell_1/ReadVariableOp_3lstm_cell_1/ReadVariableOp_32D
 lstm_cell_1/split/ReadVariableOp lstm_cell_1/split/ReadVariableOp2H
"lstm_cell_1/split_1/ReadVariableOp"lstm_cell_1/split_1/ReadVariableOp2
whilewhile:] Y
5
_output_shapes#
!:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs:VR
0
_output_shapes
:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ

_user_specified_namemask:YU
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
)
_user_specified_nameinitial_state/0:YU
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
)
_user_specified_nameinitial_state/1
í
c
G__inference_activation_1_layer_call_and_return_conditional_losses_14446

inputs
identity\
ReluReluinputs*
T0*5
_output_shapes#
!:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2
Relut
IdentityIdentityRelu:activations:0*
T0*5
_output_shapes#
!:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*4
_input_shapes#
!:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ:] Y
5
_output_shapes#
!:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs

Þ
&__inference_lstm_1_layer_call_fn_17255

inputs
mask

initial_state_0
initial_state_1
unknown
	unknown_0
	unknown_1
identity

identity_1

identity_2¢StatefulPartitionedCallá
StatefulPartitionedCallStatefulPartitionedCallinputsmaskinitial_state_0initial_state_1unknown	unknown_0	unknown_1*
Tin
	2
*
Tout
2*
_collective_manager_ids
 *]
_output_shapesK
I:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *J
fERC
A__inference_lstm_1_layer_call_and_return_conditional_losses_150512
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*5
_output_shapes#
!:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2

Identity

Identity_1Identity StatefulPartitionedCall:output:1^StatefulPartitionedCall*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity_1

Identity_2Identity StatefulPartitionedCall:output:2^StatefulPartitionedCall*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity_2"
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0*
_input_shapess
q:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:::22
StatefulPartitionedCallStatefulPartitionedCall:] Y
5
_output_shapes#
!:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs:VR
0
_output_shapes
:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ

_user_specified_namemask:YU
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
)
_user_specified_nameinitial_state/0:YU
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
)
_user_specified_nameinitial_state/1
ö$

while_body_14327
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0
while_lstm_cell_1_14351_0
while_lstm_cell_1_14353_0
while_lstm_cell_1_14355_0
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_identity_5
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor
while_lstm_cell_1_14351
while_lstm_cell_1_14353
while_lstm_cell_1_14355¢)while/lstm_cell_1/StatefulPartitionedCallÃ
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ   29
7while/TensorArrayV2Read/TensorListGetItem/element_shapeÔ
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
element_dtype02+
)while/TensorArrayV2Read/TensorListGetItemÚ
)while/lstm_cell_1/StatefulPartitionedCallStatefulPartitionedCall0while/TensorArrayV2Read/TensorListGetItem:item:0while_placeholder_2while_placeholder_3while_lstm_cell_1_14351_0while_lstm_cell_1_14353_0while_lstm_cell_1_14355_0*
Tin

2*
Tout
2*
_collective_manager_ids
 *P
_output_shapes>
<:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *O
fJRH
F__inference_lstm_cell_1_layer_call_and_return_conditional_losses_138912+
)while/lstm_cell_1/StatefulPartitionedCallö
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholder2while/lstm_cell_1/StatefulPartitionedCall:output:0*
_output_shapes
: *
element_dtype02,
*while/TensorArrayV2Write/TensorListSetItem\
while/add/yConst*
_output_shapes
: *
dtype0*
value	B :2
while/add/yi
	while/addAddV2while_placeholderwhile/add/y:output:0*
T0*
_output_shapes
: 2
	while/add`
while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :2
while/add_1/yv
while/add_1AddV2while_while_loop_counterwhile/add_1/y:output:0*
T0*
_output_shapes
: 2
while/add_1
while/IdentityIdentitywhile/add_1:z:0*^while/lstm_cell_1/StatefulPartitionedCall*
T0*
_output_shapes
: 2
while/Identity
while/Identity_1Identitywhile_while_maximum_iterations*^while/lstm_cell_1/StatefulPartitionedCall*
T0*
_output_shapes
: 2
while/Identity_1
while/Identity_2Identitywhile/add:z:0*^while/lstm_cell_1/StatefulPartitionedCall*
T0*
_output_shapes
: 2
while/Identity_2¹
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0*^while/lstm_cell_1/StatefulPartitionedCall*
T0*
_output_shapes
: 2
while/Identity_3Ã
while/Identity_4Identity2while/lstm_cell_1/StatefulPartitionedCall:output:1*^while/lstm_cell_1/StatefulPartitionedCall*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
while/Identity_4Ã
while/Identity_5Identity2while/lstm_cell_1/StatefulPartitionedCall:output:2*^while/lstm_cell_1/StatefulPartitionedCall*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
while/Identity_5")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"-
while_identity_5while/Identity_5:output:0"4
while_lstm_cell_1_14351while_lstm_cell_1_14351_0"4
while_lstm_cell_1_14353while_lstm_cell_1_14353_0"4
while_lstm_cell_1_14355while_lstm_cell_1_14355_0"0
while_strided_slice_1while_strided_slice_1_0"¨
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*S
_input_shapesB
@: : : : :ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ: : :::2V
)while/lstm_cell_1/StatefulPartitionedCall)while/lstm_cell_1/StatefulPartitionedCall: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :.*
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ:.*
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ:

_output_shapes
: :

_output_shapes
: 
æÃ

A__inference_lstm_1_layer_call_and_return_conditional_losses_16967

inputs
mask

initial_state_0
initial_state_1-
)lstm_cell_1_split_readvariableop_resource/
+lstm_cell_1_split_1_readvariableop_resource'
#lstm_cell_1_readvariableop_resource
identity

identity_1

identity_2¢lstm_cell_1/ReadVariableOp¢lstm_cell_1/ReadVariableOp_1¢lstm_cell_1/ReadVariableOp_2¢lstm_cell_1/ReadVariableOp_3¢ lstm_cell_1/split/ReadVariableOp¢"lstm_cell_1/split_1/ReadVariableOp¢whileu
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
transpose/perm
	transpose	Transposeinputstranspose/perm:output:0*
T0*5
_output_shapes#
!:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2
	transposeK
ShapeShapetranspose:y:0*
T0*
_output_shapes
:2
Shapet
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice/stackx
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_1x
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_2â
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slicek
ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ2
ExpandDims/dim

ExpandDims
ExpandDimsmaskExpandDims/dim:output:0*
T0
*4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2

ExpandDimsy
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
transpose_1/perm
transpose_1	TransposeExpandDims:output:0transpose_1/perm:output:0*
T0
*4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2
transpose_1
TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ2
TensorArrayV2/element_shape°
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
TensorArrayV2¿
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ   27
5TensorArrayUnstack/TensorListFromTensor/element_shapeø
'TensorArrayUnstack/TensorListFromTensorTensorListFromTensortranspose:y:0>TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type02)
'TensorArrayUnstack/TensorListFromTensorx
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_1/stack|
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_1|
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_2ý
strided_slice_1StridedSlicetranspose:y:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
shrink_axis_mask2
strided_slice_1y
lstm_cell_1/ones_like/ShapeShapeinitial_state_0*
T0*
_output_shapes
:2
lstm_cell_1/ones_like/Shape
lstm_cell_1/ones_like/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ?2
lstm_cell_1/ones_like/Constµ
lstm_cell_1/ones_likeFill$lstm_cell_1/ones_like/Shape:output:0$lstm_cell_1/ones_like/Const:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
lstm_cell_1/ones_like{
lstm_cell_1/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *ä8?2
lstm_cell_1/dropout/Const°
lstm_cell_1/dropout/MulMullstm_cell_1/ones_like:output:0"lstm_cell_1/dropout/Const:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
lstm_cell_1/dropout/Mul
lstm_cell_1/dropout/ShapeShapelstm_cell_1/ones_like:output:0*
T0*
_output_shapes
:2
lstm_cell_1/dropout/Shapeø
0lstm_cell_1/dropout/random_uniform/RandomUniformRandomUniform"lstm_cell_1/dropout/Shape:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
dtype0*
seed±ÿå)*
seed2¢¬Ë22
0lstm_cell_1/dropout/random_uniform/RandomUniform
"lstm_cell_1/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *ÍÌÌ=2$
"lstm_cell_1/dropout/GreaterEqual/yï
 lstm_cell_1/dropout/GreaterEqualGreaterEqual9lstm_cell_1/dropout/random_uniform/RandomUniform:output:0+lstm_cell_1/dropout/GreaterEqual/y:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2"
 lstm_cell_1/dropout/GreaterEqual¤
lstm_cell_1/dropout/CastCast$lstm_cell_1/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
lstm_cell_1/dropout/Cast«
lstm_cell_1/dropout/Mul_1Mullstm_cell_1/dropout/Mul:z:0lstm_cell_1/dropout/Cast:y:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
lstm_cell_1/dropout/Mul_1
lstm_cell_1/dropout_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *ä8?2
lstm_cell_1/dropout_1/Const¶
lstm_cell_1/dropout_1/MulMullstm_cell_1/ones_like:output:0$lstm_cell_1/dropout_1/Const:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
lstm_cell_1/dropout_1/Mul
lstm_cell_1/dropout_1/ShapeShapelstm_cell_1/ones_like:output:0*
T0*
_output_shapes
:2
lstm_cell_1/dropout_1/Shapeþ
2lstm_cell_1/dropout_1/random_uniform/RandomUniformRandomUniform$lstm_cell_1/dropout_1/Shape:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
dtype0*
seed±ÿå)*
seed2··«24
2lstm_cell_1/dropout_1/random_uniform/RandomUniform
$lstm_cell_1/dropout_1/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *ÍÌÌ=2&
$lstm_cell_1/dropout_1/GreaterEqual/y÷
"lstm_cell_1/dropout_1/GreaterEqualGreaterEqual;lstm_cell_1/dropout_1/random_uniform/RandomUniform:output:0-lstm_cell_1/dropout_1/GreaterEqual/y:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2$
"lstm_cell_1/dropout_1/GreaterEqualª
lstm_cell_1/dropout_1/CastCast&lstm_cell_1/dropout_1/GreaterEqual:z:0*

DstT0*

SrcT0
*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
lstm_cell_1/dropout_1/Cast³
lstm_cell_1/dropout_1/Mul_1Mullstm_cell_1/dropout_1/Mul:z:0lstm_cell_1/dropout_1/Cast:y:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
lstm_cell_1/dropout_1/Mul_1
lstm_cell_1/dropout_2/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *ä8?2
lstm_cell_1/dropout_2/Const¶
lstm_cell_1/dropout_2/MulMullstm_cell_1/ones_like:output:0$lstm_cell_1/dropout_2/Const:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
lstm_cell_1/dropout_2/Mul
lstm_cell_1/dropout_2/ShapeShapelstm_cell_1/ones_like:output:0*
T0*
_output_shapes
:2
lstm_cell_1/dropout_2/Shapeþ
2lstm_cell_1/dropout_2/random_uniform/RandomUniformRandomUniform$lstm_cell_1/dropout_2/Shape:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
dtype0*
seed±ÿå)*
seed2ù24
2lstm_cell_1/dropout_2/random_uniform/RandomUniform
$lstm_cell_1/dropout_2/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *ÍÌÌ=2&
$lstm_cell_1/dropout_2/GreaterEqual/y÷
"lstm_cell_1/dropout_2/GreaterEqualGreaterEqual;lstm_cell_1/dropout_2/random_uniform/RandomUniform:output:0-lstm_cell_1/dropout_2/GreaterEqual/y:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2$
"lstm_cell_1/dropout_2/GreaterEqualª
lstm_cell_1/dropout_2/CastCast&lstm_cell_1/dropout_2/GreaterEqual:z:0*

DstT0*

SrcT0
*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
lstm_cell_1/dropout_2/Cast³
lstm_cell_1/dropout_2/Mul_1Mullstm_cell_1/dropout_2/Mul:z:0lstm_cell_1/dropout_2/Cast:y:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
lstm_cell_1/dropout_2/Mul_1
lstm_cell_1/dropout_3/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *ä8?2
lstm_cell_1/dropout_3/Const¶
lstm_cell_1/dropout_3/MulMullstm_cell_1/ones_like:output:0$lstm_cell_1/dropout_3/Const:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
lstm_cell_1/dropout_3/Mul
lstm_cell_1/dropout_3/ShapeShapelstm_cell_1/ones_like:output:0*
T0*
_output_shapes
:2
lstm_cell_1/dropout_3/Shapeþ
2lstm_cell_1/dropout_3/random_uniform/RandomUniformRandomUniform$lstm_cell_1/dropout_3/Shape:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
dtype0*
seed±ÿå)*
seed2Üî24
2lstm_cell_1/dropout_3/random_uniform/RandomUniform
$lstm_cell_1/dropout_3/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *ÍÌÌ=2&
$lstm_cell_1/dropout_3/GreaterEqual/y÷
"lstm_cell_1/dropout_3/GreaterEqualGreaterEqual;lstm_cell_1/dropout_3/random_uniform/RandomUniform:output:0-lstm_cell_1/dropout_3/GreaterEqual/y:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2$
"lstm_cell_1/dropout_3/GreaterEqualª
lstm_cell_1/dropout_3/CastCast&lstm_cell_1/dropout_3/GreaterEqual:z:0*

DstT0*

SrcT0
*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
lstm_cell_1/dropout_3/Cast³
lstm_cell_1/dropout_3/Mul_1Mullstm_cell_1/dropout_3/Mul:z:0lstm_cell_1/dropout_3/Cast:y:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
lstm_cell_1/dropout_3/Mul_1h
lstm_cell_1/ConstConst*
_output_shapes
: *
dtype0*
value	B :2
lstm_cell_1/Const|
lstm_cell_1/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2
lstm_cell_1/split/split_dim°
 lstm_cell_1/split/ReadVariableOpReadVariableOp)lstm_cell_1_split_readvariableop_resource* 
_output_shapes
:
*
dtype02"
 lstm_cell_1/split/ReadVariableOpß
lstm_cell_1/splitSplit$lstm_cell_1/split/split_dim:output:0(lstm_cell_1/split/ReadVariableOp:value:0*
T0*D
_output_shapes2
0:
:
:
:
*
	num_split2
lstm_cell_1/split
lstm_cell_1/MatMulMatMulstrided_slice_1:output:0lstm_cell_1/split:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
lstm_cell_1/MatMul
lstm_cell_1/MatMul_1MatMulstrided_slice_1:output:0lstm_cell_1/split:output:1*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
lstm_cell_1/MatMul_1
lstm_cell_1/MatMul_2MatMulstrided_slice_1:output:0lstm_cell_1/split:output:2*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
lstm_cell_1/MatMul_2
lstm_cell_1/MatMul_3MatMulstrided_slice_1:output:0lstm_cell_1/split:output:3*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
lstm_cell_1/MatMul_3l
lstm_cell_1/Const_1Const*
_output_shapes
: *
dtype0*
value	B :2
lstm_cell_1/Const_1
lstm_cell_1/split_1/split_dimConst*
_output_shapes
: *
dtype0*
value	B : 2
lstm_cell_1/split_1/split_dim±
"lstm_cell_1/split_1/ReadVariableOpReadVariableOp+lstm_cell_1_split_1_readvariableop_resource*
_output_shapes	
:*
dtype02$
"lstm_cell_1/split_1/ReadVariableOpÓ
lstm_cell_1/split_1Split&lstm_cell_1/split_1/split_dim:output:0*lstm_cell_1/split_1/ReadVariableOp:value:0*
T0*0
_output_shapes
::::*
	num_split2
lstm_cell_1/split_1¤
lstm_cell_1/BiasAddBiasAddlstm_cell_1/MatMul:product:0lstm_cell_1/split_1:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
lstm_cell_1/BiasAddª
lstm_cell_1/BiasAdd_1BiasAddlstm_cell_1/MatMul_1:product:0lstm_cell_1/split_1:output:1*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
lstm_cell_1/BiasAdd_1ª
lstm_cell_1/BiasAdd_2BiasAddlstm_cell_1/MatMul_2:product:0lstm_cell_1/split_1:output:2*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
lstm_cell_1/BiasAdd_2ª
lstm_cell_1/BiasAdd_3BiasAddlstm_cell_1/MatMul_3:product:0lstm_cell_1/split_1:output:3*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
lstm_cell_1/BiasAdd_3
lstm_cell_1/mulMulinitial_state_0lstm_cell_1/dropout/Mul_1:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
lstm_cell_1/mul
lstm_cell_1/mul_1Mulinitial_state_0lstm_cell_1/dropout_1/Mul_1:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
lstm_cell_1/mul_1
lstm_cell_1/mul_2Mulinitial_state_0lstm_cell_1/dropout_2/Mul_1:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
lstm_cell_1/mul_2
lstm_cell_1/mul_3Mulinitial_state_0lstm_cell_1/dropout_3/Mul_1:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
lstm_cell_1/mul_3
lstm_cell_1/ReadVariableOpReadVariableOp#lstm_cell_1_readvariableop_resource* 
_output_shapes
:
*
dtype02
lstm_cell_1/ReadVariableOp
lstm_cell_1/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB"        2!
lstm_cell_1/strided_slice/stack
!lstm_cell_1/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       2#
!lstm_cell_1/strided_slice/stack_1
!lstm_cell_1/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2#
!lstm_cell_1/strided_slice/stack_2Æ
lstm_cell_1/strided_sliceStridedSlice"lstm_cell_1/ReadVariableOp:value:0(lstm_cell_1/strided_slice/stack:output:0*lstm_cell_1/strided_slice/stack_1:output:0*lstm_cell_1/strided_slice/stack_2:output:0*
Index0*
T0* 
_output_shapes
:
*

begin_mask*
end_mask2
lstm_cell_1/strided_slice¢
lstm_cell_1/MatMul_4MatMullstm_cell_1/mul:z:0"lstm_cell_1/strided_slice:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
lstm_cell_1/MatMul_4
lstm_cell_1/addAddV2lstm_cell_1/BiasAdd:output:0lstm_cell_1/MatMul_4:product:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
lstm_cell_1/add}
lstm_cell_1/SigmoidSigmoidlstm_cell_1/add:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
lstm_cell_1/Sigmoid¢
lstm_cell_1/ReadVariableOp_1ReadVariableOp#lstm_cell_1_readvariableop_resource* 
_output_shapes
:
*
dtype02
lstm_cell_1/ReadVariableOp_1
!lstm_cell_1/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB"       2#
!lstm_cell_1/strided_slice_1/stack
#lstm_cell_1/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       2%
#lstm_cell_1/strided_slice_1/stack_1
#lstm_cell_1/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2%
#lstm_cell_1/strided_slice_1/stack_2Ò
lstm_cell_1/strided_slice_1StridedSlice$lstm_cell_1/ReadVariableOp_1:value:0*lstm_cell_1/strided_slice_1/stack:output:0,lstm_cell_1/strided_slice_1/stack_1:output:0,lstm_cell_1/strided_slice_1/stack_2:output:0*
Index0*
T0* 
_output_shapes
:
*

begin_mask*
end_mask2
lstm_cell_1/strided_slice_1¦
lstm_cell_1/MatMul_5MatMullstm_cell_1/mul_1:z:0$lstm_cell_1/strided_slice_1:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
lstm_cell_1/MatMul_5¢
lstm_cell_1/add_1AddV2lstm_cell_1/BiasAdd_1:output:0lstm_cell_1/MatMul_5:product:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
lstm_cell_1/add_1
lstm_cell_1/Sigmoid_1Sigmoidlstm_cell_1/add_1:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
lstm_cell_1/Sigmoid_1
lstm_cell_1/mul_4Mullstm_cell_1/Sigmoid_1:y:0initial_state_1*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
lstm_cell_1/mul_4¢
lstm_cell_1/ReadVariableOp_2ReadVariableOp#lstm_cell_1_readvariableop_resource* 
_output_shapes
:
*
dtype02
lstm_cell_1/ReadVariableOp_2
!lstm_cell_1/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB"       2#
!lstm_cell_1/strided_slice_2/stack
#lstm_cell_1/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       2%
#lstm_cell_1/strided_slice_2/stack_1
#lstm_cell_1/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2%
#lstm_cell_1/strided_slice_2/stack_2Ò
lstm_cell_1/strided_slice_2StridedSlice$lstm_cell_1/ReadVariableOp_2:value:0*lstm_cell_1/strided_slice_2/stack:output:0,lstm_cell_1/strided_slice_2/stack_1:output:0,lstm_cell_1/strided_slice_2/stack_2:output:0*
Index0*
T0* 
_output_shapes
:
*

begin_mask*
end_mask2
lstm_cell_1/strided_slice_2¦
lstm_cell_1/MatMul_6MatMullstm_cell_1/mul_2:z:0$lstm_cell_1/strided_slice_2:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
lstm_cell_1/MatMul_6¢
lstm_cell_1/add_2AddV2lstm_cell_1/BiasAdd_2:output:0lstm_cell_1/MatMul_6:product:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
lstm_cell_1/add_2v
lstm_cell_1/TanhTanhlstm_cell_1/add_2:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
lstm_cell_1/Tanh
lstm_cell_1/mul_5Mullstm_cell_1/Sigmoid:y:0lstm_cell_1/Tanh:y:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
lstm_cell_1/mul_5
lstm_cell_1/add_3AddV2lstm_cell_1/mul_4:z:0lstm_cell_1/mul_5:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
lstm_cell_1/add_3¢
lstm_cell_1/ReadVariableOp_3ReadVariableOp#lstm_cell_1_readvariableop_resource* 
_output_shapes
:
*
dtype02
lstm_cell_1/ReadVariableOp_3
!lstm_cell_1/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB"       2#
!lstm_cell_1/strided_slice_3/stack
#lstm_cell_1/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        2%
#lstm_cell_1/strided_slice_3/stack_1
#lstm_cell_1/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2%
#lstm_cell_1/strided_slice_3/stack_2Ò
lstm_cell_1/strided_slice_3StridedSlice$lstm_cell_1/ReadVariableOp_3:value:0*lstm_cell_1/strided_slice_3/stack:output:0,lstm_cell_1/strided_slice_3/stack_1:output:0,lstm_cell_1/strided_slice_3/stack_2:output:0*
Index0*
T0* 
_output_shapes
:
*

begin_mask*
end_mask2
lstm_cell_1/strided_slice_3¦
lstm_cell_1/MatMul_7MatMullstm_cell_1/mul_3:z:0$lstm_cell_1/strided_slice_3:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
lstm_cell_1/MatMul_7¢
lstm_cell_1/add_4AddV2lstm_cell_1/BiasAdd_3:output:0lstm_cell_1/MatMul_7:product:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
lstm_cell_1/add_4
lstm_cell_1/Sigmoid_2Sigmoidlstm_cell_1/add_4:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
lstm_cell_1/Sigmoid_2z
lstm_cell_1/Tanh_1Tanhlstm_cell_1/add_3:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
lstm_cell_1/Tanh_1
lstm_cell_1/mul_6Mullstm_cell_1/Sigmoid_2:y:0lstm_cell_1/Tanh_1:y:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
lstm_cell_1/mul_6
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ   2
TensorArrayV2_1/element_shape¶
TensorArrayV2_1TensorListReserve&TensorArrayV2_1/element_shape:output:0strided_slice:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
TensorArrayV2_1N
timeConst*
_output_shapes
: *
dtype0*
value	B : 2
time
TensorArrayV2_2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ2
TensorArrayV2_2/element_shape¶
TensorArrayV2_2TensorListReserve&TensorArrayV2_2/element_shape:output:0strided_slice:output:0*
_output_shapes
: *
element_dtype0
*

shape_type02
TensorArrayV2_2Ã
7TensorArrayUnstack_1/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ   29
7TensorArrayUnstack_1/TensorListFromTensor/element_shape
)TensorArrayUnstack_1/TensorListFromTensorTensorListFromTensortranspose_1:y:0@TensorArrayUnstack_1/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0
*

shape_type02+
)TensorArrayUnstack_1/TensorListFromTensoro

zeros_like	ZerosLikelstm_cell_1/mul_6:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

zeros_like
while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ2
while/maximum_iterationsj
while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : 2
while/loop_counterÚ
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros_like:y:0initial_state_0initial_state_1strided_slice:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:09TensorArrayUnstack_1/TensorListFromTensor:output_handle:0)lstm_cell_1_split_readvariableop_resource+lstm_cell_1_split_1_readvariableop_resource#lstm_cell_1_readvariableop_resource*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*d
_output_shapesR
P: : : : :ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ: : : : : : *%
_read_only_resource_inputs

*
bodyR
while_body_16786*
condR
while_cond_16785*c
output_shapesR
P: : : : :ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ: : : : : : *
parallel_iterations 2
whileµ
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ   22
0TensorArrayV2Stack/TensorListStack/element_shapeò
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*5
_output_shapes#
!:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ*
element_dtype02$
"TensorArrayV2Stack/TensorListStack
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB:
ÿÿÿÿÿÿÿÿÿ2
strided_slice_2/stack|
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_2/stack_1|
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_2/stack_2
strided_slice_2StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
shrink_axis_mask2
strided_slice_2y
transpose_2/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
transpose_2/perm¯
transpose_2	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_2/perm:output:0*
T0*5
_output_shapes#
!:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2
transpose_2f
runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    2	
runtime»
IdentityIdentitytranspose_2:y:0^lstm_cell_1/ReadVariableOp^lstm_cell_1/ReadVariableOp_1^lstm_cell_1/ReadVariableOp_2^lstm_cell_1/ReadVariableOp_3!^lstm_cell_1/split/ReadVariableOp#^lstm_cell_1/split_1/ReadVariableOp^while*
T0*5
_output_shapes#
!:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2

Identity±

Identity_1Identitywhile:output:5^lstm_cell_1/ReadVariableOp^lstm_cell_1/ReadVariableOp_1^lstm_cell_1/ReadVariableOp_2^lstm_cell_1/ReadVariableOp_3!^lstm_cell_1/split/ReadVariableOp#^lstm_cell_1/split_1/ReadVariableOp^while*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity_1±

Identity_2Identitywhile:output:6^lstm_cell_1/ReadVariableOp^lstm_cell_1/ReadVariableOp_1^lstm_cell_1/ReadVariableOp_2^lstm_cell_1/ReadVariableOp_3!^lstm_cell_1/split/ReadVariableOp#^lstm_cell_1/split_1/ReadVariableOp^while*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity_2"
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0*
_input_shapess
q:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:::28
lstm_cell_1/ReadVariableOplstm_cell_1/ReadVariableOp2<
lstm_cell_1/ReadVariableOp_1lstm_cell_1/ReadVariableOp_12<
lstm_cell_1/ReadVariableOp_2lstm_cell_1/ReadVariableOp_22<
lstm_cell_1/ReadVariableOp_3lstm_cell_1/ReadVariableOp_32D
 lstm_cell_1/split/ReadVariableOp lstm_cell_1/split/ReadVariableOp2H
"lstm_cell_1/split_1/ReadVariableOp"lstm_cell_1/split_1/ReadVariableOp2
whilewhile:] Y
5
_output_shapes#
!:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs:VR
0
_output_shapes
:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ

_user_specified_namemask:YU
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
)
_user_specified_nameinitial_state/0:YU
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
)
_user_specified_nameinitial_state/1
ª
¾
while_cond_16214
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_less_strided_slice_13
/while_while_cond_16214___redundant_placeholder03
/while_while_cond_16214___redundant_placeholder13
/while_while_cond_16214___redundant_placeholder23
/while_while_cond_16214___redundant_placeholder3
while_identity
p

while/LessLesswhile_placeholderwhile_less_strided_slice_1*
T0*
_output_shapes
: 2

while/Less]
while/IdentityIdentitywhile/Less:z:0*
T0
*
_output_shapes
: 2
while/Identity")
while_identitywhile/Identity:output:0*U
_input_shapesD
B: : : : :ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ: ::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :.*
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ:.*
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ:

_output_shapes
: :

_output_shapes
:
ª
¾
while_cond_16489
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_less_strided_slice_13
/while_while_cond_16489___redundant_placeholder03
/while_while_cond_16489___redundant_placeholder13
/while_while_cond_16489___redundant_placeholder23
/while_while_cond_16489___redundant_placeholder3
while_identity
p

while/LessLesswhile_placeholderwhile_less_strided_slice_1*
T0*
_output_shapes
: 2

while/Less]
while/IdentityIdentitywhile/Less:z:0*
T0
*
_output_shapes
: 2
while/Identity")
while_identitywhile/Identity:output:0*U
_input_shapesD
B: : : : :ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ: ::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :.*
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ:.*
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ:

_output_shapes
: :

_output_shapes
:
ª
¾
while_cond_14188
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_less_strided_slice_13
/while_while_cond_14188___redundant_placeholder03
/while_while_cond_14188___redundant_placeholder13
/while_while_cond_14188___redundant_placeholder23
/while_while_cond_14188___redundant_placeholder3
while_identity
p

while/LessLesswhile_placeholderwhile_less_strided_slice_1*
T0*
_output_shapes
: 2

while/Less]
while/IdentityIdentitywhile/Less:z:0*
T0
*
_output_shapes
: 2
while/Identity")
while_identitywhile/Identity:output:0*U
_input_shapesD
B: : : : :ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ: ::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :.*
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ:.*
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ:

_output_shapes
: :

_output_shapes
:
ñ

¬
&__inference_lstm_1_layer_call_fn_16636
inputs_0
unknown
	unknown_0
	unknown_1
identity

identity_1

identity_2¢StatefulPartitionedCall¸
StatefulPartitionedCallStatefulPartitionedCallinputs_0unknown	unknown_0	unknown_1*
Tin
2*
Tout
2*
_collective_manager_ids
 *]
_output_shapesK
I:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *J
fERC
A__inference_lstm_1_layer_call_and_return_conditional_losses_142602
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*5
_output_shapes#
!:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2

Identity

Identity_1Identity StatefulPartitionedCall:output:1^StatefulPartitionedCall*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity_1

Identity_2Identity StatefulPartitionedCall:output:2^StatefulPartitionedCall*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity_2"
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0*@
_input_shapes/
-:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ:::22
StatefulPartitionedCallStatefulPartitionedCall:_ [
5
_output_shapes#
!:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
"
_user_specified_name
inputs/0
½
Å
while_body_16490
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_05
1while_lstm_cell_1_split_readvariableop_resource_07
3while_lstm_cell_1_split_1_readvariableop_resource_0/
+while_lstm_cell_1_readvariableop_resource_0
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_identity_5
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor3
/while_lstm_cell_1_split_readvariableop_resource5
1while_lstm_cell_1_split_1_readvariableop_resource-
)while_lstm_cell_1_readvariableop_resource¢ while/lstm_cell_1/ReadVariableOp¢"while/lstm_cell_1/ReadVariableOp_1¢"while/lstm_cell_1/ReadVariableOp_2¢"while/lstm_cell_1/ReadVariableOp_3¢&while/lstm_cell_1/split/ReadVariableOp¢(while/lstm_cell_1/split_1/ReadVariableOpÃ
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ   29
7while/TensorArrayV2Read/TensorListGetItem/element_shapeÔ
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
element_dtype02+
)while/TensorArrayV2Read/TensorListGetItem
!while/lstm_cell_1/ones_like/ShapeShapewhile_placeholder_2*
T0*
_output_shapes
:2#
!while/lstm_cell_1/ones_like/Shape
!while/lstm_cell_1/ones_like/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ?2#
!while/lstm_cell_1/ones_like/ConstÍ
while/lstm_cell_1/ones_likeFill*while/lstm_cell_1/ones_like/Shape:output:0*while/lstm_cell_1/ones_like/Const:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
while/lstm_cell_1/ones_liket
while/lstm_cell_1/ConstConst*
_output_shapes
: *
dtype0*
value	B :2
while/lstm_cell_1/Const
!while/lstm_cell_1/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2#
!while/lstm_cell_1/split/split_dimÄ
&while/lstm_cell_1/split/ReadVariableOpReadVariableOp1while_lstm_cell_1_split_readvariableop_resource_0* 
_output_shapes
:
*
dtype02(
&while/lstm_cell_1/split/ReadVariableOp÷
while/lstm_cell_1/splitSplit*while/lstm_cell_1/split/split_dim:output:0.while/lstm_cell_1/split/ReadVariableOp:value:0*
T0*D
_output_shapes2
0:
:
:
:
*
	num_split2
while/lstm_cell_1/splitÅ
while/lstm_cell_1/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:0 while/lstm_cell_1/split:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
while/lstm_cell_1/MatMulÉ
while/lstm_cell_1/MatMul_1MatMul0while/TensorArrayV2Read/TensorListGetItem:item:0 while/lstm_cell_1/split:output:1*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
while/lstm_cell_1/MatMul_1É
while/lstm_cell_1/MatMul_2MatMul0while/TensorArrayV2Read/TensorListGetItem:item:0 while/lstm_cell_1/split:output:2*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
while/lstm_cell_1/MatMul_2É
while/lstm_cell_1/MatMul_3MatMul0while/TensorArrayV2Read/TensorListGetItem:item:0 while/lstm_cell_1/split:output:3*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
while/lstm_cell_1/MatMul_3x
while/lstm_cell_1/Const_1Const*
_output_shapes
: *
dtype0*
value	B :2
while/lstm_cell_1/Const_1
#while/lstm_cell_1/split_1/split_dimConst*
_output_shapes
: *
dtype0*
value	B : 2%
#while/lstm_cell_1/split_1/split_dimÅ
(while/lstm_cell_1/split_1/ReadVariableOpReadVariableOp3while_lstm_cell_1_split_1_readvariableop_resource_0*
_output_shapes	
:*
dtype02*
(while/lstm_cell_1/split_1/ReadVariableOpë
while/lstm_cell_1/split_1Split,while/lstm_cell_1/split_1/split_dim:output:00while/lstm_cell_1/split_1/ReadVariableOp:value:0*
T0*0
_output_shapes
::::*
	num_split2
while/lstm_cell_1/split_1¼
while/lstm_cell_1/BiasAddBiasAdd"while/lstm_cell_1/MatMul:product:0"while/lstm_cell_1/split_1:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
while/lstm_cell_1/BiasAddÂ
while/lstm_cell_1/BiasAdd_1BiasAdd$while/lstm_cell_1/MatMul_1:product:0"while/lstm_cell_1/split_1:output:1*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
while/lstm_cell_1/BiasAdd_1Â
while/lstm_cell_1/BiasAdd_2BiasAdd$while/lstm_cell_1/MatMul_2:product:0"while/lstm_cell_1/split_1:output:2*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
while/lstm_cell_1/BiasAdd_2Â
while/lstm_cell_1/BiasAdd_3BiasAdd$while/lstm_cell_1/MatMul_3:product:0"while/lstm_cell_1/split_1:output:3*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
while/lstm_cell_1/BiasAdd_3£
while/lstm_cell_1/mulMulwhile_placeholder_2$while/lstm_cell_1/ones_like:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
while/lstm_cell_1/mul§
while/lstm_cell_1/mul_1Mulwhile_placeholder_2$while/lstm_cell_1/ones_like:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
while/lstm_cell_1/mul_1§
while/lstm_cell_1/mul_2Mulwhile_placeholder_2$while/lstm_cell_1/ones_like:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
while/lstm_cell_1/mul_2§
while/lstm_cell_1/mul_3Mulwhile_placeholder_2$while/lstm_cell_1/ones_like:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
while/lstm_cell_1/mul_3²
 while/lstm_cell_1/ReadVariableOpReadVariableOp+while_lstm_cell_1_readvariableop_resource_0* 
_output_shapes
:
*
dtype02"
 while/lstm_cell_1/ReadVariableOp
%while/lstm_cell_1/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB"        2'
%while/lstm_cell_1/strided_slice/stack£
'while/lstm_cell_1/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       2)
'while/lstm_cell_1/strided_slice/stack_1£
'while/lstm_cell_1/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2)
'while/lstm_cell_1/strided_slice/stack_2ê
while/lstm_cell_1/strided_sliceStridedSlice(while/lstm_cell_1/ReadVariableOp:value:0.while/lstm_cell_1/strided_slice/stack:output:00while/lstm_cell_1/strided_slice/stack_1:output:00while/lstm_cell_1/strided_slice/stack_2:output:0*
Index0*
T0* 
_output_shapes
:
*

begin_mask*
end_mask2!
while/lstm_cell_1/strided_sliceº
while/lstm_cell_1/MatMul_4MatMulwhile/lstm_cell_1/mul:z:0(while/lstm_cell_1/strided_slice:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
while/lstm_cell_1/MatMul_4´
while/lstm_cell_1/addAddV2"while/lstm_cell_1/BiasAdd:output:0$while/lstm_cell_1/MatMul_4:product:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
while/lstm_cell_1/add
while/lstm_cell_1/SigmoidSigmoidwhile/lstm_cell_1/add:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
while/lstm_cell_1/Sigmoid¶
"while/lstm_cell_1/ReadVariableOp_1ReadVariableOp+while_lstm_cell_1_readvariableop_resource_0* 
_output_shapes
:
*
dtype02$
"while/lstm_cell_1/ReadVariableOp_1£
'while/lstm_cell_1/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB"       2)
'while/lstm_cell_1/strided_slice_1/stack§
)while/lstm_cell_1/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       2+
)while/lstm_cell_1/strided_slice_1/stack_1§
)while/lstm_cell_1/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2+
)while/lstm_cell_1/strided_slice_1/stack_2ö
!while/lstm_cell_1/strided_slice_1StridedSlice*while/lstm_cell_1/ReadVariableOp_1:value:00while/lstm_cell_1/strided_slice_1/stack:output:02while/lstm_cell_1/strided_slice_1/stack_1:output:02while/lstm_cell_1/strided_slice_1/stack_2:output:0*
Index0*
T0* 
_output_shapes
:
*

begin_mask*
end_mask2#
!while/lstm_cell_1/strided_slice_1¾
while/lstm_cell_1/MatMul_5MatMulwhile/lstm_cell_1/mul_1:z:0*while/lstm_cell_1/strided_slice_1:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
while/lstm_cell_1/MatMul_5º
while/lstm_cell_1/add_1AddV2$while/lstm_cell_1/BiasAdd_1:output:0$while/lstm_cell_1/MatMul_5:product:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
while/lstm_cell_1/add_1
while/lstm_cell_1/Sigmoid_1Sigmoidwhile/lstm_cell_1/add_1:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
while/lstm_cell_1/Sigmoid_1¢
while/lstm_cell_1/mul_4Mulwhile/lstm_cell_1/Sigmoid_1:y:0while_placeholder_3*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
while/lstm_cell_1/mul_4¶
"while/lstm_cell_1/ReadVariableOp_2ReadVariableOp+while_lstm_cell_1_readvariableop_resource_0* 
_output_shapes
:
*
dtype02$
"while/lstm_cell_1/ReadVariableOp_2£
'while/lstm_cell_1/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB"       2)
'while/lstm_cell_1/strided_slice_2/stack§
)while/lstm_cell_1/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       2+
)while/lstm_cell_1/strided_slice_2/stack_1§
)while/lstm_cell_1/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2+
)while/lstm_cell_1/strided_slice_2/stack_2ö
!while/lstm_cell_1/strided_slice_2StridedSlice*while/lstm_cell_1/ReadVariableOp_2:value:00while/lstm_cell_1/strided_slice_2/stack:output:02while/lstm_cell_1/strided_slice_2/stack_1:output:02while/lstm_cell_1/strided_slice_2/stack_2:output:0*
Index0*
T0* 
_output_shapes
:
*

begin_mask*
end_mask2#
!while/lstm_cell_1/strided_slice_2¾
while/lstm_cell_1/MatMul_6MatMulwhile/lstm_cell_1/mul_2:z:0*while/lstm_cell_1/strided_slice_2:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
while/lstm_cell_1/MatMul_6º
while/lstm_cell_1/add_2AddV2$while/lstm_cell_1/BiasAdd_2:output:0$while/lstm_cell_1/MatMul_6:product:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
while/lstm_cell_1/add_2
while/lstm_cell_1/TanhTanhwhile/lstm_cell_1/add_2:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
while/lstm_cell_1/Tanh§
while/lstm_cell_1/mul_5Mulwhile/lstm_cell_1/Sigmoid:y:0while/lstm_cell_1/Tanh:y:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
while/lstm_cell_1/mul_5¨
while/lstm_cell_1/add_3AddV2while/lstm_cell_1/mul_4:z:0while/lstm_cell_1/mul_5:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
while/lstm_cell_1/add_3¶
"while/lstm_cell_1/ReadVariableOp_3ReadVariableOp+while_lstm_cell_1_readvariableop_resource_0* 
_output_shapes
:
*
dtype02$
"while/lstm_cell_1/ReadVariableOp_3£
'while/lstm_cell_1/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB"       2)
'while/lstm_cell_1/strided_slice_3/stack§
)while/lstm_cell_1/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        2+
)while/lstm_cell_1/strided_slice_3/stack_1§
)while/lstm_cell_1/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2+
)while/lstm_cell_1/strided_slice_3/stack_2ö
!while/lstm_cell_1/strided_slice_3StridedSlice*while/lstm_cell_1/ReadVariableOp_3:value:00while/lstm_cell_1/strided_slice_3/stack:output:02while/lstm_cell_1/strided_slice_3/stack_1:output:02while/lstm_cell_1/strided_slice_3/stack_2:output:0*
Index0*
T0* 
_output_shapes
:
*

begin_mask*
end_mask2#
!while/lstm_cell_1/strided_slice_3¾
while/lstm_cell_1/MatMul_7MatMulwhile/lstm_cell_1/mul_3:z:0*while/lstm_cell_1/strided_slice_3:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
while/lstm_cell_1/MatMul_7º
while/lstm_cell_1/add_4AddV2$while/lstm_cell_1/BiasAdd_3:output:0$while/lstm_cell_1/MatMul_7:product:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
while/lstm_cell_1/add_4
while/lstm_cell_1/Sigmoid_2Sigmoidwhile/lstm_cell_1/add_4:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
while/lstm_cell_1/Sigmoid_2
while/lstm_cell_1/Tanh_1Tanhwhile/lstm_cell_1/add_3:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
while/lstm_cell_1/Tanh_1«
while/lstm_cell_1/mul_6Mulwhile/lstm_cell_1/Sigmoid_2:y:0while/lstm_cell_1/Tanh_1:y:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
while/lstm_cell_1/mul_6ß
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholderwhile/lstm_cell_1/mul_6:z:0*
_output_shapes
: *
element_dtype02,
*while/TensorArrayV2Write/TensorListSetItem\
while/add/yConst*
_output_shapes
: *
dtype0*
value	B :2
while/add/yi
	while/addAddV2while_placeholderwhile/add/y:output:0*
T0*
_output_shapes
: 2
	while/add`
while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :2
while/add_1/yv
while/add_1AddV2while_while_loop_counterwhile/add_1/y:output:0*
T0*
_output_shapes
: 2
while/add_1Ä
while/IdentityIdentitywhile/add_1:z:0!^while/lstm_cell_1/ReadVariableOp#^while/lstm_cell_1/ReadVariableOp_1#^while/lstm_cell_1/ReadVariableOp_2#^while/lstm_cell_1/ReadVariableOp_3'^while/lstm_cell_1/split/ReadVariableOp)^while/lstm_cell_1/split_1/ReadVariableOp*
T0*
_output_shapes
: 2
while/Identity×
while/Identity_1Identitywhile_while_maximum_iterations!^while/lstm_cell_1/ReadVariableOp#^while/lstm_cell_1/ReadVariableOp_1#^while/lstm_cell_1/ReadVariableOp_2#^while/lstm_cell_1/ReadVariableOp_3'^while/lstm_cell_1/split/ReadVariableOp)^while/lstm_cell_1/split_1/ReadVariableOp*
T0*
_output_shapes
: 2
while/Identity_1Æ
while/Identity_2Identitywhile/add:z:0!^while/lstm_cell_1/ReadVariableOp#^while/lstm_cell_1/ReadVariableOp_1#^while/lstm_cell_1/ReadVariableOp_2#^while/lstm_cell_1/ReadVariableOp_3'^while/lstm_cell_1/split/ReadVariableOp)^while/lstm_cell_1/split_1/ReadVariableOp*
T0*
_output_shapes
: 2
while/Identity_2ó
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0!^while/lstm_cell_1/ReadVariableOp#^while/lstm_cell_1/ReadVariableOp_1#^while/lstm_cell_1/ReadVariableOp_2#^while/lstm_cell_1/ReadVariableOp_3'^while/lstm_cell_1/split/ReadVariableOp)^while/lstm_cell_1/split_1/ReadVariableOp*
T0*
_output_shapes
: 2
while/Identity_3æ
while/Identity_4Identitywhile/lstm_cell_1/mul_6:z:0!^while/lstm_cell_1/ReadVariableOp#^while/lstm_cell_1/ReadVariableOp_1#^while/lstm_cell_1/ReadVariableOp_2#^while/lstm_cell_1/ReadVariableOp_3'^while/lstm_cell_1/split/ReadVariableOp)^while/lstm_cell_1/split_1/ReadVariableOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
while/Identity_4æ
while/Identity_5Identitywhile/lstm_cell_1/add_3:z:0!^while/lstm_cell_1/ReadVariableOp#^while/lstm_cell_1/ReadVariableOp_1#^while/lstm_cell_1/ReadVariableOp_2#^while/lstm_cell_1/ReadVariableOp_3'^while/lstm_cell_1/split/ReadVariableOp)^while/lstm_cell_1/split_1/ReadVariableOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
while/Identity_5")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"-
while_identity_5while/Identity_5:output:0"X
)while_lstm_cell_1_readvariableop_resource+while_lstm_cell_1_readvariableop_resource_0"h
1while_lstm_cell_1_split_1_readvariableop_resource3while_lstm_cell_1_split_1_readvariableop_resource_0"d
/while_lstm_cell_1_split_readvariableop_resource1while_lstm_cell_1_split_readvariableop_resource_0"0
while_strided_slice_1while_strided_slice_1_0"¨
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*S
_input_shapesB
@: : : : :ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ: : :::2D
 while/lstm_cell_1/ReadVariableOp while/lstm_cell_1/ReadVariableOp2H
"while/lstm_cell_1/ReadVariableOp_1"while/lstm_cell_1/ReadVariableOp_12H
"while/lstm_cell_1/ReadVariableOp_2"while/lstm_cell_1/ReadVariableOp_22H
"while/lstm_cell_1/ReadVariableOp_3"while/lstm_cell_1/ReadVariableOp_32P
&while/lstm_cell_1/split/ReadVariableOp&while/lstm_cell_1/split/ReadVariableOp2T
(while/lstm_cell_1/split_1/ReadVariableOp(while/lstm_cell_1/split_1/ReadVariableOp: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :.*
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ:.*
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ:

_output_shapes
: :

_output_shapes
: 
ìG
ó
A__inference_lstm_1_layer_call_and_return_conditional_losses_14398

inputs
lstm_cell_1_14314
lstm_cell_1_14316
lstm_cell_1_14318
identity

identity_1

identity_2¢#lstm_cell_1/StatefulPartitionedCall¢whileD
ShapeShapeinputs*
T0*
_output_shapes
:2
Shapet
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice/stackx
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_1x
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_2â
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice]
zeros/mul/yConst*
_output_shapes
: *
dtype0*
value
B :2
zeros/mul/yl
	zeros/mulMulstrided_slice:output:0zeros/mul/y:output:0*
T0*
_output_shapes
: 2
	zeros/mul_
zeros/Less/yConst*
_output_shapes
: *
dtype0*
value
B :è2
zeros/Less/yg

zeros/LessLesszeros/mul:z:0zeros/Less/y:output:0*
T0*
_output_shapes
: 2

zeros/Lessc
zeros/packed/1Const*
_output_shapes
: *
dtype0*
value
B :2
zeros/packed/1
zeros/packedPackstrided_slice:output:0zeros/packed/1:output:0*
N*
T0*
_output_shapes
:2
zeros/packed_
zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
zeros/Constv
zerosFillzeros/packed:output:0zeros/Const:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
zerosa
zeros_1/mul/yConst*
_output_shapes
: *
dtype0*
value
B :2
zeros_1/mul/yr
zeros_1/mulMulstrided_slice:output:0zeros_1/mul/y:output:0*
T0*
_output_shapes
: 2
zeros_1/mulc
zeros_1/Less/yConst*
_output_shapes
: *
dtype0*
value
B :è2
zeros_1/Less/yo
zeros_1/LessLesszeros_1/mul:z:0zeros_1/Less/y:output:0*
T0*
_output_shapes
: 2
zeros_1/Lessg
zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value
B :2
zeros_1/packed/1
zeros_1/packedPackstrided_slice:output:0zeros_1/packed/1:output:0*
N*
T0*
_output_shapes
:2
zeros_1/packedc
zeros_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
zeros_1/Const~
zeros_1Fillzeros_1/packed:output:0zeros_1/Const:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2	
zeros_1u
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
transpose/perm
	transpose	Transposeinputstranspose/perm:output:0*
T0*5
_output_shapes#
!:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2
	transposeO
Shape_1Shapetranspose:y:0*
T0*
_output_shapes
:2	
Shape_1x
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_1/stack|
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_1|
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_2î
strided_slice_1StridedSliceShape_1:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice_1
TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ2
TensorArrayV2/element_shape²
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
TensorArrayV2¿
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ   27
5TensorArrayUnstack/TensorListFromTensor/element_shapeø
'TensorArrayUnstack/TensorListFromTensorTensorListFromTensortranspose:y:0>TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type02)
'TensorArrayUnstack/TensorListFromTensorx
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_2/stack|
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_2/stack_1|
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_2/stack_2ý
strided_slice_2StridedSlicetranspose:y:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
shrink_axis_mask2
strided_slice_2
#lstm_cell_1/StatefulPartitionedCallStatefulPartitionedCallstrided_slice_2:output:0zeros:output:0zeros_1:output:0lstm_cell_1_14314lstm_cell_1_14316lstm_cell_1_14318*
Tin

2*
Tout
2*
_collective_manager_ids
 *P
_output_shapes>
<:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *O
fJRH
F__inference_lstm_cell_1_layer_call_and_return_conditional_losses_138912%
#lstm_cell_1/StatefulPartitionedCall
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ   2
TensorArrayV2_1/element_shape¸
TensorArrayV2_1TensorListReserve&TensorArrayV2_1/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
TensorArrayV2_1N
timeConst*
_output_shapes
: *
dtype0*
value	B : 2
time
while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ2
while/maximum_iterationsj
while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : 2
while/loop_counter
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0zeros_1:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0lstm_cell_1_14314lstm_cell_1_14316lstm_cell_1_14318*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*N
_output_shapes<
:: : : : :ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ: : : : : *%
_read_only_resource_inputs
	
*
bodyR
while_body_14327*
condR
while_cond_14326*M
output_shapes<
:: : : : :ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ: : : : : *
parallel_iterations 2
whileµ
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ   22
0TensorArrayV2Stack/TensorListStack/element_shapeò
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*5
_output_shapes#
!:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ*
element_dtype02$
"TensorArrayV2Stack/TensorListStack
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
ÿÿÿÿÿÿÿÿÿ2
strided_slice_3/stack|
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_3/stack_1|
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_3/stack_2
strided_slice_3StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
shrink_axis_mask2
strided_slice_3y
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
transpose_1/perm¯
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*5
_output_shapes#
!:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2
transpose_1f
runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    2	
runtime
IdentityIdentitytranspose_1:y:0$^lstm_cell_1/StatefulPartitionedCall^while*
T0*5
_output_shapes#
!:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2

Identity

Identity_1Identitywhile:output:4$^lstm_cell_1/StatefulPartitionedCall^while*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity_1

Identity_2Identitywhile:output:5$^lstm_cell_1/StatefulPartitionedCall^while*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity_2"
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0*@
_input_shapes/
-:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ:::2J
#lstm_cell_1/StatefulPartitionedCall#lstm_cell_1/StatefulPartitionedCall2
whilewhile:] Y
5
_output_shapes#
!:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
êH
þ
F__inference_lstm_cell_1_layer_call_and_return_conditional_losses_13891

inputs

states
states_1!
split_readvariableop_resource#
split_1_readvariableop_resource
readvariableop_resource
identity

identity_1

identity_2¢ReadVariableOp¢ReadVariableOp_1¢ReadVariableOp_2¢ReadVariableOp_3¢split/ReadVariableOp¢split_1/ReadVariableOpX
ones_like/ShapeShapestates*
T0*
_output_shapes
:2
ones_like/Shapeg
ones_like/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ?2
ones_like/Const
	ones_likeFillones_like/Shape:output:0ones_like/Const:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
	ones_likeP
ConstConst*
_output_shapes
: *
dtype0*
value	B :2
Constd
split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2
split/split_dim
split/ReadVariableOpReadVariableOpsplit_readvariableop_resource* 
_output_shapes
:
*
dtype02
split/ReadVariableOp¯
splitSplitsplit/split_dim:output:0split/ReadVariableOp:value:0*
T0*D
_output_shapes2
0:
:
:
:
*
	num_split2
splite
MatMulMatMulinputssplit:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
MatMuli
MatMul_1MatMulinputssplit:output:1*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

MatMul_1i
MatMul_2MatMulinputssplit:output:2*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

MatMul_2i
MatMul_3MatMulinputssplit:output:3*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

MatMul_3T
Const_1Const*
_output_shapes
: *
dtype0*
value	B :2	
Const_1h
split_1/split_dimConst*
_output_shapes
: *
dtype0*
value	B : 2
split_1/split_dim
split_1/ReadVariableOpReadVariableOpsplit_1_readvariableop_resource*
_output_shapes	
:*
dtype02
split_1/ReadVariableOp£
split_1Splitsplit_1/split_dim:output:0split_1/ReadVariableOp:value:0*
T0*0
_output_shapes
::::*
	num_split2	
split_1t
BiasAddBiasAddMatMul:product:0split_1:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2	
BiasAddz
	BiasAdd_1BiasAddMatMul_1:product:0split_1:output:1*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
	BiasAdd_1z
	BiasAdd_2BiasAddMatMul_2:product:0split_1:output:2*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
	BiasAdd_2z
	BiasAdd_3BiasAddMatMul_3:product:0split_1:output:3*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
	BiasAdd_3`
mulMulstatesones_like:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
muld
mul_1Mulstatesones_like:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
mul_1d
mul_2Mulstatesones_like:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
mul_2d
mul_3Mulstatesones_like:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
mul_3z
ReadVariableOpReadVariableOpreadvariableop_resource* 
_output_shapes
:
*
dtype02
ReadVariableOp{
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB"        2
strided_slice/stack
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       2
strided_slice/stack_1
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2
strided_slice/stack_2þ
strided_sliceStridedSliceReadVariableOp:value:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0* 
_output_shapes
:
*

begin_mask*
end_mask2
strided_slicer
MatMul_4MatMulmul:z:0strided_slice:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

MatMul_4l
addAddV2BiasAdd:output:0MatMul_4:product:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
addY
SigmoidSigmoidadd:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2	
Sigmoid~
ReadVariableOp_1ReadVariableOpreadvariableop_resource* 
_output_shapes
:
*
dtype02
ReadVariableOp_1
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB"       2
strided_slice_1/stack
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       2
strided_slice_1/stack_1
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2
strided_slice_1/stack_2
strided_slice_1StridedSliceReadVariableOp_1:value:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0* 
_output_shapes
:
*

begin_mask*
end_mask2
strided_slice_1v
MatMul_5MatMul	mul_1:z:0strided_slice_1:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

MatMul_5r
add_1AddV2BiasAdd_1:output:0MatMul_5:product:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
add_1_
	Sigmoid_1Sigmoid	add_1:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
	Sigmoid_1a
mul_4MulSigmoid_1:y:0states_1*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
mul_4~
ReadVariableOp_2ReadVariableOpreadvariableop_resource* 
_output_shapes
:
*
dtype02
ReadVariableOp_2
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB"       2
strided_slice_2/stack
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       2
strided_slice_2/stack_1
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2
strided_slice_2/stack_2
strided_slice_2StridedSliceReadVariableOp_2:value:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0* 
_output_shapes
:
*

begin_mask*
end_mask2
strided_slice_2v
MatMul_6MatMul	mul_2:z:0strided_slice_2:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

MatMul_6r
add_2AddV2BiasAdd_2:output:0MatMul_6:product:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
add_2R
TanhTanh	add_2:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
Tanh_
mul_5MulSigmoid:y:0Tanh:y:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
mul_5`
add_3AddV2	mul_4:z:0	mul_5:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
add_3~
ReadVariableOp_3ReadVariableOpreadvariableop_resource* 
_output_shapes
:
*
dtype02
ReadVariableOp_3
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB"       2
strided_slice_3/stack
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        2
strided_slice_3/stack_1
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2
strided_slice_3/stack_2
strided_slice_3StridedSliceReadVariableOp_3:value:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0* 
_output_shapes
:
*

begin_mask*
end_mask2
strided_slice_3v
MatMul_7MatMul	mul_3:z:0strided_slice_3:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

MatMul_7r
add_4AddV2BiasAdd_3:output:0MatMul_7:product:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
add_4_
	Sigmoid_2Sigmoid	add_4:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
	Sigmoid_2V
Tanh_1Tanh	add_3:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
Tanh_1c
mul_6MulSigmoid_2:y:0
Tanh_1:y:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
mul_6Ø
IdentityIdentity	mul_6:z:0^ReadVariableOp^ReadVariableOp_1^ReadVariableOp_2^ReadVariableOp_3^split/ReadVariableOp^split_1/ReadVariableOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

IdentityÜ

Identity_1Identity	mul_6:z:0^ReadVariableOp^ReadVariableOp_1^ReadVariableOp_2^ReadVariableOp_3^split/ReadVariableOp^split_1/ReadVariableOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity_1Ü

Identity_2Identity	add_3:z:0^ReadVariableOp^ReadVariableOp_1^ReadVariableOp_2^ReadVariableOp_3^split/ReadVariableOp^split_1/ReadVariableOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity_2"
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0*[
_input_shapesJ
H:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:::2 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_12$
ReadVariableOp_2ReadVariableOp_22$
ReadVariableOp_3ReadVariableOp_32,
split/ReadVariableOpsplit/ReadVariableOp20
split_1/ReadVariableOpsplit_1/ReadVariableOp:P L
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs:PL
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_namestates:PL
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_namestates
Ü	

while_cond_14617
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_placeholder_4
while_less_strided_slice3
/while_while_cond_14617___redundant_placeholder03
/while_while_cond_14617___redundant_placeholder13
/while_while_cond_14617___redundant_placeholder23
/while_while_cond_14617___redundant_placeholder33
/while_while_cond_14617___redundant_placeholder4
while_identity
n

while/LessLesswhile_placeholderwhile_less_strided_slice*
T0*
_output_shapes
: 2

while/Less]
while/IdentityIdentitywhile/Less:z:0*
T0
*
_output_shapes
: 2
while/Identity")
while_identitywhile/Identity:output:0*m
_input_shapes\
Z: : : : :ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ: :::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :.*
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ:.*
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ:.*
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ:

_output_shapes
: :

_output_shapes
::	

_output_shapes
:
Ä
«
lstm_1_while_cond_15462*
&lstm_1_while_lstm_1_while_loop_counter0
,lstm_1_while_lstm_1_while_maximum_iterations
lstm_1_while_placeholder
lstm_1_while_placeholder_1
lstm_1_while_placeholder_2
lstm_1_while_placeholder_3
lstm_1_while_placeholder_4*
&lstm_1_while_less_lstm_1_strided_sliceA
=lstm_1_while_lstm_1_while_cond_15462___redundant_placeholder0A
=lstm_1_while_lstm_1_while_cond_15462___redundant_placeholder1A
=lstm_1_while_lstm_1_while_cond_15462___redundant_placeholder2A
=lstm_1_while_lstm_1_while_cond_15462___redundant_placeholder3A
=lstm_1_while_lstm_1_while_cond_15462___redundant_placeholder4
lstm_1_while_identity

lstm_1/while/LessLesslstm_1_while_placeholder&lstm_1_while_less_lstm_1_strided_slice*
T0*
_output_shapes
: 2
lstm_1/while/Lessr
lstm_1/while/IdentityIdentitylstm_1/while/Less:z:0*
T0
*
_output_shapes
: 2
lstm_1/while/Identity"7
lstm_1_while_identitylstm_1/while/Identity:output:0*m
_input_shapes\
Z: : : : :ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ: :::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :.*
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ:.*
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ:.*
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ:

_output_shapes
: :

_output_shapes
::	

_output_shapes
:
ñ

¬
&__inference_lstm_1_layer_call_fn_16651
inputs_0
unknown
	unknown_0
	unknown_1
identity

identity_1

identity_2¢StatefulPartitionedCall¸
StatefulPartitionedCallStatefulPartitionedCallinputs_0unknown	unknown_0	unknown_1*
Tin
2*
Tout
2*
_collective_manager_ids
 *]
_output_shapesK
I:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *J
fERC
A__inference_lstm_1_layer_call_and_return_conditional_losses_143982
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*5
_output_shapes#
!:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2

Identity

Identity_1Identity StatefulPartitionedCall:output:1^StatefulPartitionedCall*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity_1

Identity_2Identity StatefulPartitionedCall:output:2^StatefulPartitionedCall*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity_2"
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0*@
_input_shapes/
-:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ:::22
StatefulPartitionedCallStatefulPartitionedCall:_ [
5
_output_shapes#
!:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
"
_user_specified_name
inputs/0
À
ó
'__inference_model_2_layer_call_fn_15287
input_2
input_4
input_3
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
	unknown_4
identity

identity_1

identity_2¢StatefulPartitionedCallò
StatefulPartitionedCallStatefulPartitionedCallinput_2input_4input_3unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4*
Tin
2	*
Tout
2*
_collective_manager_ids
 *\
_output_shapesJ
H:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ7:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ*(
_read_only_resource_inputs

*-
config_proto

CPU

GPU 2J 8 *K
fFRD
B__inference_model_2_layer_call_and_return_conditional_losses_152682
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ72

Identity

Identity_1Identity StatefulPartitionedCall:output:1^StatefulPartitionedCall*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity_1

Identity_2Identity StatefulPartitionedCall:output:2^StatefulPartitionedCall*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity_2"
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0*o
_input_shapes^
\:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ::::::22
StatefulPartitionedCallStatefulPartitionedCall:Y U
0
_output_shapes
:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
!
_user_specified_name	input_2:QM
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
!
_user_specified_name	input_4:QM
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
!
_user_specified_name	input_3
ªá

B__inference_model_2_layer_call_and_return_conditional_losses_15971
inputs_0
inputs_1
inputs_2&
"embedding_1_embedding_lookup_156834
0lstm_1_lstm_cell_1_split_readvariableop_resource6
2lstm_1_lstm_cell_1_split_1_readvariableop_resource.
*lstm_1_lstm_cell_1_readvariableop_resource+
'dense_tensordot_readvariableop_resource)
%dense_biasadd_readvariableop_resource
identity

identity_1

identity_2¢dense/BiasAdd/ReadVariableOp¢dense/Tensordot/ReadVariableOp¢embedding_1/embedding_lookup¢!lstm_1/lstm_cell_1/ReadVariableOp¢#lstm_1/lstm_cell_1/ReadVariableOp_1¢#lstm_1/lstm_cell_1/ReadVariableOp_2¢#lstm_1/lstm_cell_1/ReadVariableOp_3¢'lstm_1/lstm_cell_1/split/ReadVariableOp¢)lstm_1/lstm_cell_1/split_1/ReadVariableOp¢lstm_1/while
embedding_1/CastCastinputs_0*

DstT0*

SrcT0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2
embedding_1/CastÁ
embedding_1/embedding_lookupResourceGather"embedding_1_embedding_lookup_15683embedding_1/Cast:y:0",/job:localhost/replica:0/task:0/device:CPU:0*
Tindices0*5
_class+
)'loc:@embedding_1/embedding_lookup/15683*5
_output_shapes#
!:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ*
dtype02
embedding_1/embedding_lookup¦
%embedding_1/embedding_lookup/IdentityIdentity%embedding_1/embedding_lookup:output:0",/job:localhost/replica:0/task:0/device:CPU:0*
T0*5
_class+
)'loc:@embedding_1/embedding_lookup/15683*5
_output_shapes#
!:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2'
%embedding_1/embedding_lookup/IdentityÎ
'embedding_1/embedding_lookup/Identity_1Identity.embedding_1/embedding_lookup/Identity:output:0*
T0*5
_output_shapes#
!:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2)
'embedding_1/embedding_lookup/Identity_1u
embedding_1/NotEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *    2
embedding_1/NotEqual/y
embedding_1/NotEqualNotEqualinputs_0embedding_1/NotEqual/y:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2
embedding_1/NotEqual 
activation_1/ReluRelu0embedding_1/embedding_lookup/Identity_1:output:0*
T0*5
_output_shapes#
!:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2
activation_1/Relu
dropout_1/IdentityIdentityactivation_1/Relu:activations:0*
T0*5
_output_shapes#
!:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2
dropout_1/Identity
lstm_1/transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
lstm_1/transpose/perm®
lstm_1/transpose	Transposedropout_1/Identity:output:0lstm_1/transpose/perm:output:0*
T0*5
_output_shapes#
!:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2
lstm_1/transpose`
lstm_1/ShapeShapelstm_1/transpose:y:0*
T0*
_output_shapes
:2
lstm_1/Shape
lstm_1/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
lstm_1/strided_slice/stack
lstm_1/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
lstm_1/strided_slice/stack_1
lstm_1/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
lstm_1/strided_slice/stack_2
lstm_1/strided_sliceStridedSlicelstm_1/Shape:output:0#lstm_1/strided_slice/stack:output:0%lstm_1/strided_slice/stack_1:output:0%lstm_1/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
lstm_1/strided_slicey
lstm_1/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ2
lstm_1/ExpandDims/dim­
lstm_1/ExpandDims
ExpandDimsembedding_1/NotEqual:z:0lstm_1/ExpandDims/dim:output:0*
T0
*4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2
lstm_1/ExpandDims
lstm_1/transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
lstm_1/transpose_1/perm²
lstm_1/transpose_1	Transposelstm_1/ExpandDims:output:0 lstm_1/transpose_1/perm:output:0*
T0
*4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2
lstm_1/transpose_1
"lstm_1/TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ2$
"lstm_1/TensorArrayV2/element_shapeÌ
lstm_1/TensorArrayV2TensorListReserve+lstm_1/TensorArrayV2/element_shape:output:0lstm_1/strided_slice:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
lstm_1/TensorArrayV2Í
<lstm_1/TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ   2>
<lstm_1/TensorArrayUnstack/TensorListFromTensor/element_shape
.lstm_1/TensorArrayUnstack/TensorListFromTensorTensorListFromTensorlstm_1/transpose:y:0Elstm_1/TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type020
.lstm_1/TensorArrayUnstack/TensorListFromTensor
lstm_1/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
lstm_1/strided_slice_1/stack
lstm_1/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2 
lstm_1/strided_slice_1/stack_1
lstm_1/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2 
lstm_1/strided_slice_1/stack_2§
lstm_1/strided_slice_1StridedSlicelstm_1/transpose:y:0%lstm_1/strided_slice_1/stack:output:0'lstm_1/strided_slice_1/stack_1:output:0'lstm_1/strided_slice_1/stack_2:output:0*
Index0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
shrink_axis_mask2
lstm_1/strided_slice_1
"lstm_1/lstm_cell_1/ones_like/ShapeShapeinputs_1*
T0*
_output_shapes
:2$
"lstm_1/lstm_cell_1/ones_like/Shape
"lstm_1/lstm_cell_1/ones_like/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ?2$
"lstm_1/lstm_cell_1/ones_like/ConstÑ
lstm_1/lstm_cell_1/ones_likeFill+lstm_1/lstm_cell_1/ones_like/Shape:output:0+lstm_1/lstm_cell_1/ones_like/Const:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
lstm_1/lstm_cell_1/ones_likev
lstm_1/lstm_cell_1/ConstConst*
_output_shapes
: *
dtype0*
value	B :2
lstm_1/lstm_cell_1/Const
"lstm_1/lstm_cell_1/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2$
"lstm_1/lstm_cell_1/split/split_dimÅ
'lstm_1/lstm_cell_1/split/ReadVariableOpReadVariableOp0lstm_1_lstm_cell_1_split_readvariableop_resource* 
_output_shapes
:
*
dtype02)
'lstm_1/lstm_cell_1/split/ReadVariableOpû
lstm_1/lstm_cell_1/splitSplit+lstm_1/lstm_cell_1/split/split_dim:output:0/lstm_1/lstm_cell_1/split/ReadVariableOp:value:0*
T0*D
_output_shapes2
0:
:
:
:
*
	num_split2
lstm_1/lstm_cell_1/split·
lstm_1/lstm_cell_1/MatMulMatMullstm_1/strided_slice_1:output:0!lstm_1/lstm_cell_1/split:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
lstm_1/lstm_cell_1/MatMul»
lstm_1/lstm_cell_1/MatMul_1MatMullstm_1/strided_slice_1:output:0!lstm_1/lstm_cell_1/split:output:1*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
lstm_1/lstm_cell_1/MatMul_1»
lstm_1/lstm_cell_1/MatMul_2MatMullstm_1/strided_slice_1:output:0!lstm_1/lstm_cell_1/split:output:2*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
lstm_1/lstm_cell_1/MatMul_2»
lstm_1/lstm_cell_1/MatMul_3MatMullstm_1/strided_slice_1:output:0!lstm_1/lstm_cell_1/split:output:3*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
lstm_1/lstm_cell_1/MatMul_3z
lstm_1/lstm_cell_1/Const_1Const*
_output_shapes
: *
dtype0*
value	B :2
lstm_1/lstm_cell_1/Const_1
$lstm_1/lstm_cell_1/split_1/split_dimConst*
_output_shapes
: *
dtype0*
value	B : 2&
$lstm_1/lstm_cell_1/split_1/split_dimÆ
)lstm_1/lstm_cell_1/split_1/ReadVariableOpReadVariableOp2lstm_1_lstm_cell_1_split_1_readvariableop_resource*
_output_shapes	
:*
dtype02+
)lstm_1/lstm_cell_1/split_1/ReadVariableOpï
lstm_1/lstm_cell_1/split_1Split-lstm_1/lstm_cell_1/split_1/split_dim:output:01lstm_1/lstm_cell_1/split_1/ReadVariableOp:value:0*
T0*0
_output_shapes
::::*
	num_split2
lstm_1/lstm_cell_1/split_1À
lstm_1/lstm_cell_1/BiasAddBiasAdd#lstm_1/lstm_cell_1/MatMul:product:0#lstm_1/lstm_cell_1/split_1:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
lstm_1/lstm_cell_1/BiasAddÆ
lstm_1/lstm_cell_1/BiasAdd_1BiasAdd%lstm_1/lstm_cell_1/MatMul_1:product:0#lstm_1/lstm_cell_1/split_1:output:1*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
lstm_1/lstm_cell_1/BiasAdd_1Æ
lstm_1/lstm_cell_1/BiasAdd_2BiasAdd%lstm_1/lstm_cell_1/MatMul_2:product:0#lstm_1/lstm_cell_1/split_1:output:2*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
lstm_1/lstm_cell_1/BiasAdd_2Æ
lstm_1/lstm_cell_1/BiasAdd_3BiasAdd%lstm_1/lstm_cell_1/MatMul_3:product:0#lstm_1/lstm_cell_1/split_1:output:3*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
lstm_1/lstm_cell_1/BiasAdd_3
lstm_1/lstm_cell_1/mulMulinputs_1%lstm_1/lstm_cell_1/ones_like:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
lstm_1/lstm_cell_1/mul
lstm_1/lstm_cell_1/mul_1Mulinputs_1%lstm_1/lstm_cell_1/ones_like:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
lstm_1/lstm_cell_1/mul_1
lstm_1/lstm_cell_1/mul_2Mulinputs_1%lstm_1/lstm_cell_1/ones_like:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
lstm_1/lstm_cell_1/mul_2
lstm_1/lstm_cell_1/mul_3Mulinputs_1%lstm_1/lstm_cell_1/ones_like:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
lstm_1/lstm_cell_1/mul_3³
!lstm_1/lstm_cell_1/ReadVariableOpReadVariableOp*lstm_1_lstm_cell_1_readvariableop_resource* 
_output_shapes
:
*
dtype02#
!lstm_1/lstm_cell_1/ReadVariableOp¡
&lstm_1/lstm_cell_1/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB"        2(
&lstm_1/lstm_cell_1/strided_slice/stack¥
(lstm_1/lstm_cell_1/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       2*
(lstm_1/lstm_cell_1/strided_slice/stack_1¥
(lstm_1/lstm_cell_1/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2*
(lstm_1/lstm_cell_1/strided_slice/stack_2ð
 lstm_1/lstm_cell_1/strided_sliceStridedSlice)lstm_1/lstm_cell_1/ReadVariableOp:value:0/lstm_1/lstm_cell_1/strided_slice/stack:output:01lstm_1/lstm_cell_1/strided_slice/stack_1:output:01lstm_1/lstm_cell_1/strided_slice/stack_2:output:0*
Index0*
T0* 
_output_shapes
:
*

begin_mask*
end_mask2"
 lstm_1/lstm_cell_1/strided_slice¾
lstm_1/lstm_cell_1/MatMul_4MatMullstm_1/lstm_cell_1/mul:z:0)lstm_1/lstm_cell_1/strided_slice:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
lstm_1/lstm_cell_1/MatMul_4¸
lstm_1/lstm_cell_1/addAddV2#lstm_1/lstm_cell_1/BiasAdd:output:0%lstm_1/lstm_cell_1/MatMul_4:product:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
lstm_1/lstm_cell_1/add
lstm_1/lstm_cell_1/SigmoidSigmoidlstm_1/lstm_cell_1/add:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
lstm_1/lstm_cell_1/Sigmoid·
#lstm_1/lstm_cell_1/ReadVariableOp_1ReadVariableOp*lstm_1_lstm_cell_1_readvariableop_resource* 
_output_shapes
:
*
dtype02%
#lstm_1/lstm_cell_1/ReadVariableOp_1¥
(lstm_1/lstm_cell_1/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB"       2*
(lstm_1/lstm_cell_1/strided_slice_1/stack©
*lstm_1/lstm_cell_1/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       2,
*lstm_1/lstm_cell_1/strided_slice_1/stack_1©
*lstm_1/lstm_cell_1/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2,
*lstm_1/lstm_cell_1/strided_slice_1/stack_2ü
"lstm_1/lstm_cell_1/strided_slice_1StridedSlice+lstm_1/lstm_cell_1/ReadVariableOp_1:value:01lstm_1/lstm_cell_1/strided_slice_1/stack:output:03lstm_1/lstm_cell_1/strided_slice_1/stack_1:output:03lstm_1/lstm_cell_1/strided_slice_1/stack_2:output:0*
Index0*
T0* 
_output_shapes
:
*

begin_mask*
end_mask2$
"lstm_1/lstm_cell_1/strided_slice_1Â
lstm_1/lstm_cell_1/MatMul_5MatMullstm_1/lstm_cell_1/mul_1:z:0+lstm_1/lstm_cell_1/strided_slice_1:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
lstm_1/lstm_cell_1/MatMul_5¾
lstm_1/lstm_cell_1/add_1AddV2%lstm_1/lstm_cell_1/BiasAdd_1:output:0%lstm_1/lstm_cell_1/MatMul_5:product:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
lstm_1/lstm_cell_1/add_1
lstm_1/lstm_cell_1/Sigmoid_1Sigmoidlstm_1/lstm_cell_1/add_1:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
lstm_1/lstm_cell_1/Sigmoid_1
lstm_1/lstm_cell_1/mul_4Mul lstm_1/lstm_cell_1/Sigmoid_1:y:0inputs_2*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
lstm_1/lstm_cell_1/mul_4·
#lstm_1/lstm_cell_1/ReadVariableOp_2ReadVariableOp*lstm_1_lstm_cell_1_readvariableop_resource* 
_output_shapes
:
*
dtype02%
#lstm_1/lstm_cell_1/ReadVariableOp_2¥
(lstm_1/lstm_cell_1/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB"       2*
(lstm_1/lstm_cell_1/strided_slice_2/stack©
*lstm_1/lstm_cell_1/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       2,
*lstm_1/lstm_cell_1/strided_slice_2/stack_1©
*lstm_1/lstm_cell_1/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2,
*lstm_1/lstm_cell_1/strided_slice_2/stack_2ü
"lstm_1/lstm_cell_1/strided_slice_2StridedSlice+lstm_1/lstm_cell_1/ReadVariableOp_2:value:01lstm_1/lstm_cell_1/strided_slice_2/stack:output:03lstm_1/lstm_cell_1/strided_slice_2/stack_1:output:03lstm_1/lstm_cell_1/strided_slice_2/stack_2:output:0*
Index0*
T0* 
_output_shapes
:
*

begin_mask*
end_mask2$
"lstm_1/lstm_cell_1/strided_slice_2Â
lstm_1/lstm_cell_1/MatMul_6MatMullstm_1/lstm_cell_1/mul_2:z:0+lstm_1/lstm_cell_1/strided_slice_2:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
lstm_1/lstm_cell_1/MatMul_6¾
lstm_1/lstm_cell_1/add_2AddV2%lstm_1/lstm_cell_1/BiasAdd_2:output:0%lstm_1/lstm_cell_1/MatMul_6:product:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
lstm_1/lstm_cell_1/add_2
lstm_1/lstm_cell_1/TanhTanhlstm_1/lstm_cell_1/add_2:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
lstm_1/lstm_cell_1/Tanh«
lstm_1/lstm_cell_1/mul_5Mullstm_1/lstm_cell_1/Sigmoid:y:0lstm_1/lstm_cell_1/Tanh:y:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
lstm_1/lstm_cell_1/mul_5¬
lstm_1/lstm_cell_1/add_3AddV2lstm_1/lstm_cell_1/mul_4:z:0lstm_1/lstm_cell_1/mul_5:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
lstm_1/lstm_cell_1/add_3·
#lstm_1/lstm_cell_1/ReadVariableOp_3ReadVariableOp*lstm_1_lstm_cell_1_readvariableop_resource* 
_output_shapes
:
*
dtype02%
#lstm_1/lstm_cell_1/ReadVariableOp_3¥
(lstm_1/lstm_cell_1/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB"       2*
(lstm_1/lstm_cell_1/strided_slice_3/stack©
*lstm_1/lstm_cell_1/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        2,
*lstm_1/lstm_cell_1/strided_slice_3/stack_1©
*lstm_1/lstm_cell_1/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2,
*lstm_1/lstm_cell_1/strided_slice_3/stack_2ü
"lstm_1/lstm_cell_1/strided_slice_3StridedSlice+lstm_1/lstm_cell_1/ReadVariableOp_3:value:01lstm_1/lstm_cell_1/strided_slice_3/stack:output:03lstm_1/lstm_cell_1/strided_slice_3/stack_1:output:03lstm_1/lstm_cell_1/strided_slice_3/stack_2:output:0*
Index0*
T0* 
_output_shapes
:
*

begin_mask*
end_mask2$
"lstm_1/lstm_cell_1/strided_slice_3Â
lstm_1/lstm_cell_1/MatMul_7MatMullstm_1/lstm_cell_1/mul_3:z:0+lstm_1/lstm_cell_1/strided_slice_3:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
lstm_1/lstm_cell_1/MatMul_7¾
lstm_1/lstm_cell_1/add_4AddV2%lstm_1/lstm_cell_1/BiasAdd_3:output:0%lstm_1/lstm_cell_1/MatMul_7:product:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
lstm_1/lstm_cell_1/add_4
lstm_1/lstm_cell_1/Sigmoid_2Sigmoidlstm_1/lstm_cell_1/add_4:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
lstm_1/lstm_cell_1/Sigmoid_2
lstm_1/lstm_cell_1/Tanh_1Tanhlstm_1/lstm_cell_1/add_3:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
lstm_1/lstm_cell_1/Tanh_1¯
lstm_1/lstm_cell_1/mul_6Mul lstm_1/lstm_cell_1/Sigmoid_2:y:0lstm_1/lstm_cell_1/Tanh_1:y:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
lstm_1/lstm_cell_1/mul_6
$lstm_1/TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ   2&
$lstm_1/TensorArrayV2_1/element_shapeÒ
lstm_1/TensorArrayV2_1TensorListReserve-lstm_1/TensorArrayV2_1/element_shape:output:0lstm_1/strided_slice:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
lstm_1/TensorArrayV2_1\
lstm_1/timeConst*
_output_shapes
: *
dtype0*
value	B : 2
lstm_1/time
$lstm_1/TensorArrayV2_2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ2&
$lstm_1/TensorArrayV2_2/element_shapeÒ
lstm_1/TensorArrayV2_2TensorListReserve-lstm_1/TensorArrayV2_2/element_shape:output:0lstm_1/strided_slice:output:0*
_output_shapes
: *
element_dtype0
*

shape_type02
lstm_1/TensorArrayV2_2Ñ
>lstm_1/TensorArrayUnstack_1/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ   2@
>lstm_1/TensorArrayUnstack_1/TensorListFromTensor/element_shape
0lstm_1/TensorArrayUnstack_1/TensorListFromTensorTensorListFromTensorlstm_1/transpose_1:y:0Glstm_1/TensorArrayUnstack_1/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0
*

shape_type022
0lstm_1/TensorArrayUnstack_1/TensorListFromTensor
lstm_1/zeros_like	ZerosLikelstm_1/lstm_cell_1/mul_6:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
lstm_1/zeros_like
lstm_1/while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ2!
lstm_1/while/maximum_iterationsx
lstm_1/while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : 2
lstm_1/while/loop_counterµ
lstm_1/whileWhile"lstm_1/while/loop_counter:output:0(lstm_1/while/maximum_iterations:output:0lstm_1/time:output:0lstm_1/TensorArrayV2_1:handle:0lstm_1/zeros_like:y:0inputs_1inputs_2lstm_1/strided_slice:output:0>lstm_1/TensorArrayUnstack/TensorListFromTensor:output_handle:0@lstm_1/TensorArrayUnstack_1/TensorListFromTensor:output_handle:00lstm_1_lstm_cell_1_split_readvariableop_resource2lstm_1_lstm_cell_1_split_1_readvariableop_resource*lstm_1_lstm_cell_1_readvariableop_resource*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*d
_output_shapesR
P: : : : :ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ: : : : : : *%
_read_only_resource_inputs

*#
bodyR
lstm_1_while_body_15789*#
condR
lstm_1_while_cond_15788*c
output_shapesR
P: : : : :ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ: : : : : : *
parallel_iterations 2
lstm_1/whileÃ
7lstm_1/TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ   29
7lstm_1/TensorArrayV2Stack/TensorListStack/element_shape
)lstm_1/TensorArrayV2Stack/TensorListStackTensorListStacklstm_1/while:output:3@lstm_1/TensorArrayV2Stack/TensorListStack/element_shape:output:0*5
_output_shapes#
!:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ*
element_dtype02+
)lstm_1/TensorArrayV2Stack/TensorListStack
lstm_1/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB:
ÿÿÿÿÿÿÿÿÿ2
lstm_1/strided_slice_2/stack
lstm_1/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2 
lstm_1/strided_slice_2/stack_1
lstm_1/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2 
lstm_1/strided_slice_2/stack_2Å
lstm_1/strided_slice_2StridedSlice2lstm_1/TensorArrayV2Stack/TensorListStack:tensor:0%lstm_1/strided_slice_2/stack:output:0'lstm_1/strided_slice_2/stack_1:output:0'lstm_1/strided_slice_2/stack_2:output:0*
Index0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
shrink_axis_mask2
lstm_1/strided_slice_2
lstm_1/transpose_2/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
lstm_1/transpose_2/permË
lstm_1/transpose_2	Transpose2lstm_1/TensorArrayV2Stack/TensorListStack:tensor:0 lstm_1/transpose_2/perm:output:0*
T0*5
_output_shapes#
!:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2
lstm_1/transpose_2t
lstm_1/runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    2
lstm_1/runtime©
dense/Tensordot/ReadVariableOpReadVariableOp'dense_tensordot_readvariableop_resource*
_output_shapes
:	7*
dtype02 
dense/Tensordot/ReadVariableOpv
dense/Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:2
dense/Tensordot/axes}
dense/Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       2
dense/Tensordot/freet
dense/Tensordot/ShapeShapelstm_1/transpose_2:y:0*
T0*
_output_shapes
:2
dense/Tensordot/Shape
dense/Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : 2
dense/Tensordot/GatherV2/axisï
dense/Tensordot/GatherV2GatherV2dense/Tensordot/Shape:output:0dense/Tensordot/free:output:0&dense/Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2
dense/Tensordot/GatherV2
dense/Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2!
dense/Tensordot/GatherV2_1/axisõ
dense/Tensordot/GatherV2_1GatherV2dense/Tensordot/Shape:output:0dense/Tensordot/axes:output:0(dense/Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2
dense/Tensordot/GatherV2_1x
dense/Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2
dense/Tensordot/Const
dense/Tensordot/ProdProd!dense/Tensordot/GatherV2:output:0dense/Tensordot/Const:output:0*
T0*
_output_shapes
: 2
dense/Tensordot/Prod|
dense/Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: 2
dense/Tensordot/Const_1 
dense/Tensordot/Prod_1Prod#dense/Tensordot/GatherV2_1:output:0 dense/Tensordot/Const_1:output:0*
T0*
_output_shapes
: 2
dense/Tensordot/Prod_1|
dense/Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 2
dense/Tensordot/concat/axisÎ
dense/Tensordot/concatConcatV2dense/Tensordot/free:output:0dense/Tensordot/axes:output:0$dense/Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:2
dense/Tensordot/concat¤
dense/Tensordot/stackPackdense/Tensordot/Prod:output:0dense/Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:2
dense/Tensordot/stack¼
dense/Tensordot/transpose	Transposelstm_1/transpose_2:y:0dense/Tensordot/concat:output:0*
T0*5
_output_shapes#
!:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2
dense/Tensordot/transpose·
dense/Tensordot/ReshapeReshapedense/Tensordot/transpose:y:0dense/Tensordot/stack:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2
dense/Tensordot/Reshape¶
dense/Tensordot/MatMulMatMul dense/Tensordot/Reshape:output:0&dense/Tensordot/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ72
dense/Tensordot/MatMul|
dense/Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:72
dense/Tensordot/Const_2
dense/Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2
dense/Tensordot/concat_1/axisÛ
dense/Tensordot/concat_1ConcatV2!dense/Tensordot/GatherV2:output:0 dense/Tensordot/Const_2:output:0&dense/Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:2
dense/Tensordot/concat_1±
dense/TensordotReshape dense/Tensordot/MatMul:product:0!dense/Tensordot/concat_1:output:0*
T0*4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ72
dense/Tensordot
dense/BiasAdd/ReadVariableOpReadVariableOp%dense_biasadd_readvariableop_resource*
_output_shapes
:7*
dtype02
dense/BiasAdd/ReadVariableOp¨
dense/BiasAddBiasAdddense/Tensordot:output:0$dense/BiasAdd/ReadVariableOp:value:0*
T0*4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ72
dense/BiasAdd
dense/Max/reduction_indicesConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ2
dense/Max/reduction_indices«
	dense/MaxMaxdense/BiasAdd:output:0$dense/Max/reduction_indices:output:0*
T0*4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ*
	keep_dims(2
	dense/Max
	dense/subSubdense/BiasAdd:output:0dense/Max:output:0*
T0*4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ72
	dense/subk
	dense/ExpExpdense/sub:z:0*
T0*4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ72
	dense/Exp
dense/Sum/reduction_indicesConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ2
dense/Sum/reduction_indices¢
	dense/SumSumdense/Exp:y:0$dense/Sum/reduction_indices:output:0*
T0*4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ*
	keep_dims(2
	dense/Sum
dense/truedivRealDivdense/Exp:y:0dense/Sum:output:0*
T0*4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ72
dense/truedivÌ
IdentityIdentitydense/truediv:z:0^dense/BiasAdd/ReadVariableOp^dense/Tensordot/ReadVariableOp^embedding_1/embedding_lookup"^lstm_1/lstm_cell_1/ReadVariableOp$^lstm_1/lstm_cell_1/ReadVariableOp_1$^lstm_1/lstm_cell_1/ReadVariableOp_2$^lstm_1/lstm_cell_1/ReadVariableOp_3(^lstm_1/lstm_cell_1/split/ReadVariableOp*^lstm_1/lstm_cell_1/split_1/ReadVariableOp^lstm_1/while*
T0*4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ72

IdentityÈ

Identity_1Identitylstm_1/while:output:5^dense/BiasAdd/ReadVariableOp^dense/Tensordot/ReadVariableOp^embedding_1/embedding_lookup"^lstm_1/lstm_cell_1/ReadVariableOp$^lstm_1/lstm_cell_1/ReadVariableOp_1$^lstm_1/lstm_cell_1/ReadVariableOp_2$^lstm_1/lstm_cell_1/ReadVariableOp_3(^lstm_1/lstm_cell_1/split/ReadVariableOp*^lstm_1/lstm_cell_1/split_1/ReadVariableOp^lstm_1/while*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity_1È

Identity_2Identitylstm_1/while:output:6^dense/BiasAdd/ReadVariableOp^dense/Tensordot/ReadVariableOp^embedding_1/embedding_lookup"^lstm_1/lstm_cell_1/ReadVariableOp$^lstm_1/lstm_cell_1/ReadVariableOp_1$^lstm_1/lstm_cell_1/ReadVariableOp_2$^lstm_1/lstm_cell_1/ReadVariableOp_3(^lstm_1/lstm_cell_1/split/ReadVariableOp*^lstm_1/lstm_cell_1/split_1/ReadVariableOp^lstm_1/while*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity_2"
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0*o
_input_shapes^
\:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ::::::2<
dense/BiasAdd/ReadVariableOpdense/BiasAdd/ReadVariableOp2@
dense/Tensordot/ReadVariableOpdense/Tensordot/ReadVariableOp2<
embedding_1/embedding_lookupembedding_1/embedding_lookup2F
!lstm_1/lstm_cell_1/ReadVariableOp!lstm_1/lstm_cell_1/ReadVariableOp2J
#lstm_1/lstm_cell_1/ReadVariableOp_1#lstm_1/lstm_cell_1/ReadVariableOp_12J
#lstm_1/lstm_cell_1/ReadVariableOp_2#lstm_1/lstm_cell_1/ReadVariableOp_22J
#lstm_1/lstm_cell_1/ReadVariableOp_3#lstm_1/lstm_cell_1/ReadVariableOp_32R
'lstm_1/lstm_cell_1/split/ReadVariableOp'lstm_1/lstm_cell_1/split/ReadVariableOp2V
)lstm_1/lstm_cell_1/split_1/ReadVariableOp)lstm_1/lstm_cell_1/split_1/ReadVariableOp2
lstm_1/whilelstm_1/while:Z V
0
_output_shapes
:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
"
_user_specified_name
inputs/0:RN
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
"
_user_specified_name
inputs/1:RN
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
"
_user_specified_name
inputs/2
Ê
E
)__inference_dropout_1_layer_call_fn_16071

inputs
identityÐ
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *5
_output_shapes#
!:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *M
fHRF
D__inference_dropout_1_layer_call_and_return_conditional_losses_144712
PartitionedCallz
IdentityIdentityPartitionedCall:output:0*
T0*5
_output_shapes#
!:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*4
_input_shapes#
!:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ:] Y
5
_output_shapes#
!:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
û
¨

while_body_17070
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_placeholder_4
while_strided_slice_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0[
Wwhile_tensorarrayv2read_1_tensorlistgetitem_tensorarrayunstack_1_tensorlistfromtensor_05
1while_lstm_cell_1_split_readvariableop_resource_07
3while_lstm_cell_1_split_1_readvariableop_resource_0/
+while_lstm_cell_1_readvariableop_resource_0
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_identity_5
while_identity_6
while_strided_sliceU
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorY
Uwhile_tensorarrayv2read_1_tensorlistgetitem_tensorarrayunstack_1_tensorlistfromtensor3
/while_lstm_cell_1_split_readvariableop_resource5
1while_lstm_cell_1_split_1_readvariableop_resource-
)while_lstm_cell_1_readvariableop_resource¢ while/lstm_cell_1/ReadVariableOp¢"while/lstm_cell_1/ReadVariableOp_1¢"while/lstm_cell_1/ReadVariableOp_2¢"while/lstm_cell_1/ReadVariableOp_3¢&while/lstm_cell_1/split/ReadVariableOp¢(while/lstm_cell_1/split_1/ReadVariableOpÃ
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ   29
7while/TensorArrayV2Read/TensorListGetItem/element_shapeÔ
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
element_dtype02+
)while/TensorArrayV2Read/TensorListGetItemÇ
9while/TensorArrayV2Read_1/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ   2;
9while/TensorArrayV2Read_1/TensorListGetItem/element_shapeÝ
+while/TensorArrayV2Read_1/TensorListGetItemTensorListGetItemWwhile_tensorarrayv2read_1_tensorlistgetitem_tensorarrayunstack_1_tensorlistfromtensor_0while_placeholderBwhile/TensorArrayV2Read_1/TensorListGetItem/element_shape:output:0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
element_dtype0
2-
+while/TensorArrayV2Read_1/TensorListGetItem
!while/lstm_cell_1/ones_like/ShapeShapewhile_placeholder_3*
T0*
_output_shapes
:2#
!while/lstm_cell_1/ones_like/Shape
!while/lstm_cell_1/ones_like/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ?2#
!while/lstm_cell_1/ones_like/ConstÍ
while/lstm_cell_1/ones_likeFill*while/lstm_cell_1/ones_like/Shape:output:0*while/lstm_cell_1/ones_like/Const:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
while/lstm_cell_1/ones_liket
while/lstm_cell_1/ConstConst*
_output_shapes
: *
dtype0*
value	B :2
while/lstm_cell_1/Const
!while/lstm_cell_1/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2#
!while/lstm_cell_1/split/split_dimÄ
&while/lstm_cell_1/split/ReadVariableOpReadVariableOp1while_lstm_cell_1_split_readvariableop_resource_0* 
_output_shapes
:
*
dtype02(
&while/lstm_cell_1/split/ReadVariableOp÷
while/lstm_cell_1/splitSplit*while/lstm_cell_1/split/split_dim:output:0.while/lstm_cell_1/split/ReadVariableOp:value:0*
T0*D
_output_shapes2
0:
:
:
:
*
	num_split2
while/lstm_cell_1/splitÅ
while/lstm_cell_1/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:0 while/lstm_cell_1/split:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
while/lstm_cell_1/MatMulÉ
while/lstm_cell_1/MatMul_1MatMul0while/TensorArrayV2Read/TensorListGetItem:item:0 while/lstm_cell_1/split:output:1*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
while/lstm_cell_1/MatMul_1É
while/lstm_cell_1/MatMul_2MatMul0while/TensorArrayV2Read/TensorListGetItem:item:0 while/lstm_cell_1/split:output:2*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
while/lstm_cell_1/MatMul_2É
while/lstm_cell_1/MatMul_3MatMul0while/TensorArrayV2Read/TensorListGetItem:item:0 while/lstm_cell_1/split:output:3*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
while/lstm_cell_1/MatMul_3x
while/lstm_cell_1/Const_1Const*
_output_shapes
: *
dtype0*
value	B :2
while/lstm_cell_1/Const_1
#while/lstm_cell_1/split_1/split_dimConst*
_output_shapes
: *
dtype0*
value	B : 2%
#while/lstm_cell_1/split_1/split_dimÅ
(while/lstm_cell_1/split_1/ReadVariableOpReadVariableOp3while_lstm_cell_1_split_1_readvariableop_resource_0*
_output_shapes	
:*
dtype02*
(while/lstm_cell_1/split_1/ReadVariableOpë
while/lstm_cell_1/split_1Split,while/lstm_cell_1/split_1/split_dim:output:00while/lstm_cell_1/split_1/ReadVariableOp:value:0*
T0*0
_output_shapes
::::*
	num_split2
while/lstm_cell_1/split_1¼
while/lstm_cell_1/BiasAddBiasAdd"while/lstm_cell_1/MatMul:product:0"while/lstm_cell_1/split_1:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
while/lstm_cell_1/BiasAddÂ
while/lstm_cell_1/BiasAdd_1BiasAdd$while/lstm_cell_1/MatMul_1:product:0"while/lstm_cell_1/split_1:output:1*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
while/lstm_cell_1/BiasAdd_1Â
while/lstm_cell_1/BiasAdd_2BiasAdd$while/lstm_cell_1/MatMul_2:product:0"while/lstm_cell_1/split_1:output:2*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
while/lstm_cell_1/BiasAdd_2Â
while/lstm_cell_1/BiasAdd_3BiasAdd$while/lstm_cell_1/MatMul_3:product:0"while/lstm_cell_1/split_1:output:3*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
while/lstm_cell_1/BiasAdd_3£
while/lstm_cell_1/mulMulwhile_placeholder_3$while/lstm_cell_1/ones_like:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
while/lstm_cell_1/mul§
while/lstm_cell_1/mul_1Mulwhile_placeholder_3$while/lstm_cell_1/ones_like:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
while/lstm_cell_1/mul_1§
while/lstm_cell_1/mul_2Mulwhile_placeholder_3$while/lstm_cell_1/ones_like:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
while/lstm_cell_1/mul_2§
while/lstm_cell_1/mul_3Mulwhile_placeholder_3$while/lstm_cell_1/ones_like:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
while/lstm_cell_1/mul_3²
 while/lstm_cell_1/ReadVariableOpReadVariableOp+while_lstm_cell_1_readvariableop_resource_0* 
_output_shapes
:
*
dtype02"
 while/lstm_cell_1/ReadVariableOp
%while/lstm_cell_1/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB"        2'
%while/lstm_cell_1/strided_slice/stack£
'while/lstm_cell_1/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       2)
'while/lstm_cell_1/strided_slice/stack_1£
'while/lstm_cell_1/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2)
'while/lstm_cell_1/strided_slice/stack_2ê
while/lstm_cell_1/strided_sliceStridedSlice(while/lstm_cell_1/ReadVariableOp:value:0.while/lstm_cell_1/strided_slice/stack:output:00while/lstm_cell_1/strided_slice/stack_1:output:00while/lstm_cell_1/strided_slice/stack_2:output:0*
Index0*
T0* 
_output_shapes
:
*

begin_mask*
end_mask2!
while/lstm_cell_1/strided_sliceº
while/lstm_cell_1/MatMul_4MatMulwhile/lstm_cell_1/mul:z:0(while/lstm_cell_1/strided_slice:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
while/lstm_cell_1/MatMul_4´
while/lstm_cell_1/addAddV2"while/lstm_cell_1/BiasAdd:output:0$while/lstm_cell_1/MatMul_4:product:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
while/lstm_cell_1/add
while/lstm_cell_1/SigmoidSigmoidwhile/lstm_cell_1/add:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
while/lstm_cell_1/Sigmoid¶
"while/lstm_cell_1/ReadVariableOp_1ReadVariableOp+while_lstm_cell_1_readvariableop_resource_0* 
_output_shapes
:
*
dtype02$
"while/lstm_cell_1/ReadVariableOp_1£
'while/lstm_cell_1/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB"       2)
'while/lstm_cell_1/strided_slice_1/stack§
)while/lstm_cell_1/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       2+
)while/lstm_cell_1/strided_slice_1/stack_1§
)while/lstm_cell_1/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2+
)while/lstm_cell_1/strided_slice_1/stack_2ö
!while/lstm_cell_1/strided_slice_1StridedSlice*while/lstm_cell_1/ReadVariableOp_1:value:00while/lstm_cell_1/strided_slice_1/stack:output:02while/lstm_cell_1/strided_slice_1/stack_1:output:02while/lstm_cell_1/strided_slice_1/stack_2:output:0*
Index0*
T0* 
_output_shapes
:
*

begin_mask*
end_mask2#
!while/lstm_cell_1/strided_slice_1¾
while/lstm_cell_1/MatMul_5MatMulwhile/lstm_cell_1/mul_1:z:0*while/lstm_cell_1/strided_slice_1:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
while/lstm_cell_1/MatMul_5º
while/lstm_cell_1/add_1AddV2$while/lstm_cell_1/BiasAdd_1:output:0$while/lstm_cell_1/MatMul_5:product:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
while/lstm_cell_1/add_1
while/lstm_cell_1/Sigmoid_1Sigmoidwhile/lstm_cell_1/add_1:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
while/lstm_cell_1/Sigmoid_1¢
while/lstm_cell_1/mul_4Mulwhile/lstm_cell_1/Sigmoid_1:y:0while_placeholder_4*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
while/lstm_cell_1/mul_4¶
"while/lstm_cell_1/ReadVariableOp_2ReadVariableOp+while_lstm_cell_1_readvariableop_resource_0* 
_output_shapes
:
*
dtype02$
"while/lstm_cell_1/ReadVariableOp_2£
'while/lstm_cell_1/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB"       2)
'while/lstm_cell_1/strided_slice_2/stack§
)while/lstm_cell_1/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       2+
)while/lstm_cell_1/strided_slice_2/stack_1§
)while/lstm_cell_1/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2+
)while/lstm_cell_1/strided_slice_2/stack_2ö
!while/lstm_cell_1/strided_slice_2StridedSlice*while/lstm_cell_1/ReadVariableOp_2:value:00while/lstm_cell_1/strided_slice_2/stack:output:02while/lstm_cell_1/strided_slice_2/stack_1:output:02while/lstm_cell_1/strided_slice_2/stack_2:output:0*
Index0*
T0* 
_output_shapes
:
*

begin_mask*
end_mask2#
!while/lstm_cell_1/strided_slice_2¾
while/lstm_cell_1/MatMul_6MatMulwhile/lstm_cell_1/mul_2:z:0*while/lstm_cell_1/strided_slice_2:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
while/lstm_cell_1/MatMul_6º
while/lstm_cell_1/add_2AddV2$while/lstm_cell_1/BiasAdd_2:output:0$while/lstm_cell_1/MatMul_6:product:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
while/lstm_cell_1/add_2
while/lstm_cell_1/TanhTanhwhile/lstm_cell_1/add_2:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
while/lstm_cell_1/Tanh§
while/lstm_cell_1/mul_5Mulwhile/lstm_cell_1/Sigmoid:y:0while/lstm_cell_1/Tanh:y:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
while/lstm_cell_1/mul_5¨
while/lstm_cell_1/add_3AddV2while/lstm_cell_1/mul_4:z:0while/lstm_cell_1/mul_5:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
while/lstm_cell_1/add_3¶
"while/lstm_cell_1/ReadVariableOp_3ReadVariableOp+while_lstm_cell_1_readvariableop_resource_0* 
_output_shapes
:
*
dtype02$
"while/lstm_cell_1/ReadVariableOp_3£
'while/lstm_cell_1/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB"       2)
'while/lstm_cell_1/strided_slice_3/stack§
)while/lstm_cell_1/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        2+
)while/lstm_cell_1/strided_slice_3/stack_1§
)while/lstm_cell_1/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2+
)while/lstm_cell_1/strided_slice_3/stack_2ö
!while/lstm_cell_1/strided_slice_3StridedSlice*while/lstm_cell_1/ReadVariableOp_3:value:00while/lstm_cell_1/strided_slice_3/stack:output:02while/lstm_cell_1/strided_slice_3/stack_1:output:02while/lstm_cell_1/strided_slice_3/stack_2:output:0*
Index0*
T0* 
_output_shapes
:
*

begin_mask*
end_mask2#
!while/lstm_cell_1/strided_slice_3¾
while/lstm_cell_1/MatMul_7MatMulwhile/lstm_cell_1/mul_3:z:0*while/lstm_cell_1/strided_slice_3:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
while/lstm_cell_1/MatMul_7º
while/lstm_cell_1/add_4AddV2$while/lstm_cell_1/BiasAdd_3:output:0$while/lstm_cell_1/MatMul_7:product:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
while/lstm_cell_1/add_4
while/lstm_cell_1/Sigmoid_2Sigmoidwhile/lstm_cell_1/add_4:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
while/lstm_cell_1/Sigmoid_2
while/lstm_cell_1/Tanh_1Tanhwhile/lstm_cell_1/add_3:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
while/lstm_cell_1/Tanh_1«
while/lstm_cell_1/mul_6Mulwhile/lstm_cell_1/Sigmoid_2:y:0while/lstm_cell_1/Tanh_1:y:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
while/lstm_cell_1/mul_6}
while/Tile/multiplesConst*
_output_shapes
:*
dtype0*
valueB"      2
while/Tile/multiples¥

while/TileTile2while/TensorArrayV2Read_1/TensorListGetItem:item:0while/Tile/multiples:output:0*
T0
*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

while/Tile¦
while/SelectV2SelectV2while/Tile:output:0while/lstm_cell_1/mul_6:z:0while_placeholder_2*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
while/SelectV2
while/Tile_1/multiplesConst*
_output_shapes
:*
dtype0*
valueB"      2
while/Tile_1/multiples«
while/Tile_1Tile2while/TensorArrayV2Read_1/TensorListGetItem:item:0while/Tile_1/multiples:output:0*
T0
*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
while/Tile_1
while/Tile_2/multiplesConst*
_output_shapes
:*
dtype0*
valueB"      2
while/Tile_2/multiples«
while/Tile_2Tile2while/TensorArrayV2Read_1/TensorListGetItem:item:0while/Tile_2/multiples:output:0*
T0
*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
while/Tile_2¬
while/SelectV2_1SelectV2while/Tile_1:output:0while/lstm_cell_1/mul_6:z:0while_placeholder_3*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
while/SelectV2_1¬
while/SelectV2_2SelectV2while/Tile_2:output:0while/lstm_cell_1/add_3:z:0while_placeholder_4*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
while/SelectV2_2Û
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholderwhile/SelectV2:output:0*
_output_shapes
: *
element_dtype02,
*while/TensorArrayV2Write/TensorListSetItem\
while/add/yConst*
_output_shapes
: *
dtype0*
value	B :2
while/add/yi
	while/addAddV2while_placeholderwhile/add/y:output:0*
T0*
_output_shapes
: 2
	while/add`
while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :2
while/add_1/yv
while/add_1AddV2while_while_loop_counterwhile/add_1/y:output:0*
T0*
_output_shapes
: 2
while/add_1Ä
while/IdentityIdentitywhile/add_1:z:0!^while/lstm_cell_1/ReadVariableOp#^while/lstm_cell_1/ReadVariableOp_1#^while/lstm_cell_1/ReadVariableOp_2#^while/lstm_cell_1/ReadVariableOp_3'^while/lstm_cell_1/split/ReadVariableOp)^while/lstm_cell_1/split_1/ReadVariableOp*
T0*
_output_shapes
: 2
while/Identity×
while/Identity_1Identitywhile_while_maximum_iterations!^while/lstm_cell_1/ReadVariableOp#^while/lstm_cell_1/ReadVariableOp_1#^while/lstm_cell_1/ReadVariableOp_2#^while/lstm_cell_1/ReadVariableOp_3'^while/lstm_cell_1/split/ReadVariableOp)^while/lstm_cell_1/split_1/ReadVariableOp*
T0*
_output_shapes
: 2
while/Identity_1Æ
while/Identity_2Identitywhile/add:z:0!^while/lstm_cell_1/ReadVariableOp#^while/lstm_cell_1/ReadVariableOp_1#^while/lstm_cell_1/ReadVariableOp_2#^while/lstm_cell_1/ReadVariableOp_3'^while/lstm_cell_1/split/ReadVariableOp)^while/lstm_cell_1/split_1/ReadVariableOp*
T0*
_output_shapes
: 2
while/Identity_2ó
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0!^while/lstm_cell_1/ReadVariableOp#^while/lstm_cell_1/ReadVariableOp_1#^while/lstm_cell_1/ReadVariableOp_2#^while/lstm_cell_1/ReadVariableOp_3'^while/lstm_cell_1/split/ReadVariableOp)^while/lstm_cell_1/split_1/ReadVariableOp*
T0*
_output_shapes
: 2
while/Identity_3â
while/Identity_4Identitywhile/SelectV2:output:0!^while/lstm_cell_1/ReadVariableOp#^while/lstm_cell_1/ReadVariableOp_1#^while/lstm_cell_1/ReadVariableOp_2#^while/lstm_cell_1/ReadVariableOp_3'^while/lstm_cell_1/split/ReadVariableOp)^while/lstm_cell_1/split_1/ReadVariableOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
while/Identity_4ä
while/Identity_5Identitywhile/SelectV2_1:output:0!^while/lstm_cell_1/ReadVariableOp#^while/lstm_cell_1/ReadVariableOp_1#^while/lstm_cell_1/ReadVariableOp_2#^while/lstm_cell_1/ReadVariableOp_3'^while/lstm_cell_1/split/ReadVariableOp)^while/lstm_cell_1/split_1/ReadVariableOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
while/Identity_5ä
while/Identity_6Identitywhile/SelectV2_2:output:0!^while/lstm_cell_1/ReadVariableOp#^while/lstm_cell_1/ReadVariableOp_1#^while/lstm_cell_1/ReadVariableOp_2#^while/lstm_cell_1/ReadVariableOp_3'^while/lstm_cell_1/split/ReadVariableOp)^while/lstm_cell_1/split_1/ReadVariableOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
while/Identity_6")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"-
while_identity_5while/Identity_5:output:0"-
while_identity_6while/Identity_6:output:0"X
)while_lstm_cell_1_readvariableop_resource+while_lstm_cell_1_readvariableop_resource_0"h
1while_lstm_cell_1_split_1_readvariableop_resource3while_lstm_cell_1_split_1_readvariableop_resource_0"d
/while_lstm_cell_1_split_readvariableop_resource1while_lstm_cell_1_split_readvariableop_resource_0",
while_strided_slicewhile_strided_slice_0"°
Uwhile_tensorarrayv2read_1_tensorlistgetitem_tensorarrayunstack_1_tensorlistfromtensorWwhile_tensorarrayv2read_1_tensorlistgetitem_tensorarrayunstack_1_tensorlistfromtensor_0"¨
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*i
_input_shapesX
V: : : : :ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ: : : :::2D
 while/lstm_cell_1/ReadVariableOp while/lstm_cell_1/ReadVariableOp2H
"while/lstm_cell_1/ReadVariableOp_1"while/lstm_cell_1/ReadVariableOp_12H
"while/lstm_cell_1/ReadVariableOp_2"while/lstm_cell_1/ReadVariableOp_22H
"while/lstm_cell_1/ReadVariableOp_3"while/lstm_cell_1/ReadVariableOp_32P
&while/lstm_cell_1/split/ReadVariableOp&while/lstm_cell_1/split/ReadVariableOp2T
(while/lstm_cell_1/split_1/ReadVariableOp(while/lstm_cell_1/split_1/ReadVariableOp: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :.*
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ:.*
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ:.*
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ:

_output_shapes
: :

_output_shapes
: :	

_output_shapes
: 
Â
Ö
!__inference__traced_restore_17594
file_prefix+
'assignvariableop_embedding_1_embeddings#
assignvariableop_1_dense_kernel!
assignvariableop_2_dense_bias0
,assignvariableop_3_lstm_1_lstm_cell_1_kernel:
6assignvariableop_4_lstm_1_lstm_cell_1_recurrent_kernel.
*assignvariableop_5_lstm_1_lstm_cell_1_bias

identity_7¢AssignVariableOp¢AssignVariableOp_1¢AssignVariableOp_2¢AssignVariableOp_3¢AssignVariableOp_4¢AssignVariableOp_5É
RestoreV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:*
dtype0*Õ
valueËBÈB:layer_with_weights-0/embeddings/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUEB&variables/1/.ATTRIBUTES/VARIABLE_VALUEB&variables/2/.ATTRIBUTES/VARIABLE_VALUEB&variables/3/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH2
RestoreV2/tensor_names
RestoreV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:*
dtype0*!
valueBB B B B B B B 2
RestoreV2/shape_and_slicesÎ
	RestoreV2	RestoreV2file_prefixRestoreV2/tensor_names:output:0#RestoreV2/shape_and_slices:output:0"/device:CPU:0*0
_output_shapes
:::::::*
dtypes
	22
	RestoreV2g
IdentityIdentityRestoreV2:tensors:0"/device:CPU:0*
T0*
_output_shapes
:2

Identity¦
AssignVariableOpAssignVariableOp'assignvariableop_embedding_1_embeddingsIdentity:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOpk

Identity_1IdentityRestoreV2:tensors:1"/device:CPU:0*
T0*
_output_shapes
:2

Identity_1¤
AssignVariableOp_1AssignVariableOpassignvariableop_1_dense_kernelIdentity_1:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_1k

Identity_2IdentityRestoreV2:tensors:2"/device:CPU:0*
T0*
_output_shapes
:2

Identity_2¢
AssignVariableOp_2AssignVariableOpassignvariableop_2_dense_biasIdentity_2:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_2k

Identity_3IdentityRestoreV2:tensors:3"/device:CPU:0*
T0*
_output_shapes
:2

Identity_3±
AssignVariableOp_3AssignVariableOp,assignvariableop_3_lstm_1_lstm_cell_1_kernelIdentity_3:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_3k

Identity_4IdentityRestoreV2:tensors:4"/device:CPU:0*
T0*
_output_shapes
:2

Identity_4»
AssignVariableOp_4AssignVariableOp6assignvariableop_4_lstm_1_lstm_cell_1_recurrent_kernelIdentity_4:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_4k

Identity_5IdentityRestoreV2:tensors:5"/device:CPU:0*
T0*
_output_shapes
:2

Identity_5¯
AssignVariableOp_5AssignVariableOp*assignvariableop_5_lstm_1_lstm_cell_1_biasIdentity_5:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_59
NoOpNoOp"/device:CPU:0*
_output_shapes
 2
NoOpä

Identity_6Identityfile_prefix^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_2^AssignVariableOp_3^AssignVariableOp_4^AssignVariableOp_5^NoOp"/device:CPU:0*
T0*
_output_shapes
: 2

Identity_6Ö

Identity_7IdentityIdentity_6:output:0^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_2^AssignVariableOp_3^AssignVariableOp_4^AssignVariableOp_5*
T0*
_output_shapes
: 2

Identity_7"!

identity_7Identity_7:output:0*-
_input_shapes
: ::::::2$
AssignVariableOpAssignVariableOp2(
AssignVariableOp_1AssignVariableOp_12(
AssignVariableOp_2AssignVariableOp_22(
AssignVariableOp_3AssignVariableOp_32(
AssignVariableOp_4AssignVariableOp_42(
AssignVariableOp_5AssignVariableOp_5:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix
Ï

model_2_lstm_1_while_body_13483:
6model_2_lstm_1_while_model_2_lstm_1_while_loop_counter@
<model_2_lstm_1_while_model_2_lstm_1_while_maximum_iterations$
 model_2_lstm_1_while_placeholder&
"model_2_lstm_1_while_placeholder_1&
"model_2_lstm_1_while_placeholder_2&
"model_2_lstm_1_while_placeholder_3&
"model_2_lstm_1_while_placeholder_47
3model_2_lstm_1_while_model_2_lstm_1_strided_slice_0u
qmodel_2_lstm_1_while_tensorarrayv2read_tensorlistgetitem_model_2_lstm_1_tensorarrayunstack_tensorlistfromtensor_0y
umodel_2_lstm_1_while_tensorarrayv2read_1_tensorlistgetitem_model_2_lstm_1_tensorarrayunstack_1_tensorlistfromtensor_0D
@model_2_lstm_1_while_lstm_cell_1_split_readvariableop_resource_0F
Bmodel_2_lstm_1_while_lstm_cell_1_split_1_readvariableop_resource_0>
:model_2_lstm_1_while_lstm_cell_1_readvariableop_resource_0!
model_2_lstm_1_while_identity#
model_2_lstm_1_while_identity_1#
model_2_lstm_1_while_identity_2#
model_2_lstm_1_while_identity_3#
model_2_lstm_1_while_identity_4#
model_2_lstm_1_while_identity_5#
model_2_lstm_1_while_identity_65
1model_2_lstm_1_while_model_2_lstm_1_strided_slices
omodel_2_lstm_1_while_tensorarrayv2read_tensorlistgetitem_model_2_lstm_1_tensorarrayunstack_tensorlistfromtensorw
smodel_2_lstm_1_while_tensorarrayv2read_1_tensorlistgetitem_model_2_lstm_1_tensorarrayunstack_1_tensorlistfromtensorB
>model_2_lstm_1_while_lstm_cell_1_split_readvariableop_resourceD
@model_2_lstm_1_while_lstm_cell_1_split_1_readvariableop_resource<
8model_2_lstm_1_while_lstm_cell_1_readvariableop_resource¢/model_2/lstm_1/while/lstm_cell_1/ReadVariableOp¢1model_2/lstm_1/while/lstm_cell_1/ReadVariableOp_1¢1model_2/lstm_1/while/lstm_cell_1/ReadVariableOp_2¢1model_2/lstm_1/while/lstm_cell_1/ReadVariableOp_3¢5model_2/lstm_1/while/lstm_cell_1/split/ReadVariableOp¢7model_2/lstm_1/while/lstm_cell_1/split_1/ReadVariableOpá
Fmodel_2/lstm_1/while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ   2H
Fmodel_2/lstm_1/while/TensorArrayV2Read/TensorListGetItem/element_shape®
8model_2/lstm_1/while/TensorArrayV2Read/TensorListGetItemTensorListGetItemqmodel_2_lstm_1_while_tensorarrayv2read_tensorlistgetitem_model_2_lstm_1_tensorarrayunstack_tensorlistfromtensor_0 model_2_lstm_1_while_placeholderOmodel_2/lstm_1/while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
element_dtype02:
8model_2/lstm_1/while/TensorArrayV2Read/TensorListGetItemå
Hmodel_2/lstm_1/while/TensorArrayV2Read_1/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ   2J
Hmodel_2/lstm_1/while/TensorArrayV2Read_1/TensorListGetItem/element_shape·
:model_2/lstm_1/while/TensorArrayV2Read_1/TensorListGetItemTensorListGetItemumodel_2_lstm_1_while_tensorarrayv2read_1_tensorlistgetitem_model_2_lstm_1_tensorarrayunstack_1_tensorlistfromtensor_0 model_2_lstm_1_while_placeholderQmodel_2/lstm_1/while/TensorArrayV2Read_1/TensorListGetItem/element_shape:output:0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
element_dtype0
2<
:model_2/lstm_1/while/TensorArrayV2Read_1/TensorListGetItem¶
0model_2/lstm_1/while/lstm_cell_1/ones_like/ShapeShape"model_2_lstm_1_while_placeholder_3*
T0*
_output_shapes
:22
0model_2/lstm_1/while/lstm_cell_1/ones_like/Shape©
0model_2/lstm_1/while/lstm_cell_1/ones_like/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ?22
0model_2/lstm_1/while/lstm_cell_1/ones_like/Const
*model_2/lstm_1/while/lstm_cell_1/ones_likeFill9model_2/lstm_1/while/lstm_cell_1/ones_like/Shape:output:09model_2/lstm_1/while/lstm_cell_1/ones_like/Const:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2,
*model_2/lstm_1/while/lstm_cell_1/ones_like
&model_2/lstm_1/while/lstm_cell_1/ConstConst*
_output_shapes
: *
dtype0*
value	B :2(
&model_2/lstm_1/while/lstm_cell_1/Const¦
0model_2/lstm_1/while/lstm_cell_1/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :22
0model_2/lstm_1/while/lstm_cell_1/split/split_dimñ
5model_2/lstm_1/while/lstm_cell_1/split/ReadVariableOpReadVariableOp@model_2_lstm_1_while_lstm_cell_1_split_readvariableop_resource_0* 
_output_shapes
:
*
dtype027
5model_2/lstm_1/while/lstm_cell_1/split/ReadVariableOp³
&model_2/lstm_1/while/lstm_cell_1/splitSplit9model_2/lstm_1/while/lstm_cell_1/split/split_dim:output:0=model_2/lstm_1/while/lstm_cell_1/split/ReadVariableOp:value:0*
T0*D
_output_shapes2
0:
:
:
:
*
	num_split2(
&model_2/lstm_1/while/lstm_cell_1/split
'model_2/lstm_1/while/lstm_cell_1/MatMulMatMul?model_2/lstm_1/while/TensorArrayV2Read/TensorListGetItem:item:0/model_2/lstm_1/while/lstm_cell_1/split:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2)
'model_2/lstm_1/while/lstm_cell_1/MatMul
)model_2/lstm_1/while/lstm_cell_1/MatMul_1MatMul?model_2/lstm_1/while/TensorArrayV2Read/TensorListGetItem:item:0/model_2/lstm_1/while/lstm_cell_1/split:output:1*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2+
)model_2/lstm_1/while/lstm_cell_1/MatMul_1
)model_2/lstm_1/while/lstm_cell_1/MatMul_2MatMul?model_2/lstm_1/while/TensorArrayV2Read/TensorListGetItem:item:0/model_2/lstm_1/while/lstm_cell_1/split:output:2*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2+
)model_2/lstm_1/while/lstm_cell_1/MatMul_2
)model_2/lstm_1/while/lstm_cell_1/MatMul_3MatMul?model_2/lstm_1/while/TensorArrayV2Read/TensorListGetItem:item:0/model_2/lstm_1/while/lstm_cell_1/split:output:3*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2+
)model_2/lstm_1/while/lstm_cell_1/MatMul_3
(model_2/lstm_1/while/lstm_cell_1/Const_1Const*
_output_shapes
: *
dtype0*
value	B :2*
(model_2/lstm_1/while/lstm_cell_1/Const_1ª
2model_2/lstm_1/while/lstm_cell_1/split_1/split_dimConst*
_output_shapes
: *
dtype0*
value	B : 24
2model_2/lstm_1/while/lstm_cell_1/split_1/split_dimò
7model_2/lstm_1/while/lstm_cell_1/split_1/ReadVariableOpReadVariableOpBmodel_2_lstm_1_while_lstm_cell_1_split_1_readvariableop_resource_0*
_output_shapes	
:*
dtype029
7model_2/lstm_1/while/lstm_cell_1/split_1/ReadVariableOp§
(model_2/lstm_1/while/lstm_cell_1/split_1Split;model_2/lstm_1/while/lstm_cell_1/split_1/split_dim:output:0?model_2/lstm_1/while/lstm_cell_1/split_1/ReadVariableOp:value:0*
T0*0
_output_shapes
::::*
	num_split2*
(model_2/lstm_1/while/lstm_cell_1/split_1ø
(model_2/lstm_1/while/lstm_cell_1/BiasAddBiasAdd1model_2/lstm_1/while/lstm_cell_1/MatMul:product:01model_2/lstm_1/while/lstm_cell_1/split_1:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2*
(model_2/lstm_1/while/lstm_cell_1/BiasAddþ
*model_2/lstm_1/while/lstm_cell_1/BiasAdd_1BiasAdd3model_2/lstm_1/while/lstm_cell_1/MatMul_1:product:01model_2/lstm_1/while/lstm_cell_1/split_1:output:1*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2,
*model_2/lstm_1/while/lstm_cell_1/BiasAdd_1þ
*model_2/lstm_1/while/lstm_cell_1/BiasAdd_2BiasAdd3model_2/lstm_1/while/lstm_cell_1/MatMul_2:product:01model_2/lstm_1/while/lstm_cell_1/split_1:output:2*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2,
*model_2/lstm_1/while/lstm_cell_1/BiasAdd_2þ
*model_2/lstm_1/while/lstm_cell_1/BiasAdd_3BiasAdd3model_2/lstm_1/while/lstm_cell_1/MatMul_3:product:01model_2/lstm_1/while/lstm_cell_1/split_1:output:3*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2,
*model_2/lstm_1/while/lstm_cell_1/BiasAdd_3ß
$model_2/lstm_1/while/lstm_cell_1/mulMul"model_2_lstm_1_while_placeholder_33model_2/lstm_1/while/lstm_cell_1/ones_like:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2&
$model_2/lstm_1/while/lstm_cell_1/mulã
&model_2/lstm_1/while/lstm_cell_1/mul_1Mul"model_2_lstm_1_while_placeholder_33model_2/lstm_1/while/lstm_cell_1/ones_like:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2(
&model_2/lstm_1/while/lstm_cell_1/mul_1ã
&model_2/lstm_1/while/lstm_cell_1/mul_2Mul"model_2_lstm_1_while_placeholder_33model_2/lstm_1/while/lstm_cell_1/ones_like:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2(
&model_2/lstm_1/while/lstm_cell_1/mul_2ã
&model_2/lstm_1/while/lstm_cell_1/mul_3Mul"model_2_lstm_1_while_placeholder_33model_2/lstm_1/while/lstm_cell_1/ones_like:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2(
&model_2/lstm_1/while/lstm_cell_1/mul_3ß
/model_2/lstm_1/while/lstm_cell_1/ReadVariableOpReadVariableOp:model_2_lstm_1_while_lstm_cell_1_readvariableop_resource_0* 
_output_shapes
:
*
dtype021
/model_2/lstm_1/while/lstm_cell_1/ReadVariableOp½
4model_2/lstm_1/while/lstm_cell_1/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB"        26
4model_2/lstm_1/while/lstm_cell_1/strided_slice/stackÁ
6model_2/lstm_1/while/lstm_cell_1/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       28
6model_2/lstm_1/while/lstm_cell_1/strided_slice/stack_1Á
6model_2/lstm_1/while/lstm_cell_1/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      28
6model_2/lstm_1/while/lstm_cell_1/strided_slice/stack_2Ä
.model_2/lstm_1/while/lstm_cell_1/strided_sliceStridedSlice7model_2/lstm_1/while/lstm_cell_1/ReadVariableOp:value:0=model_2/lstm_1/while/lstm_cell_1/strided_slice/stack:output:0?model_2/lstm_1/while/lstm_cell_1/strided_slice/stack_1:output:0?model_2/lstm_1/while/lstm_cell_1/strided_slice/stack_2:output:0*
Index0*
T0* 
_output_shapes
:
*

begin_mask*
end_mask20
.model_2/lstm_1/while/lstm_cell_1/strided_sliceö
)model_2/lstm_1/while/lstm_cell_1/MatMul_4MatMul(model_2/lstm_1/while/lstm_cell_1/mul:z:07model_2/lstm_1/while/lstm_cell_1/strided_slice:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2+
)model_2/lstm_1/while/lstm_cell_1/MatMul_4ð
$model_2/lstm_1/while/lstm_cell_1/addAddV21model_2/lstm_1/while/lstm_cell_1/BiasAdd:output:03model_2/lstm_1/while/lstm_cell_1/MatMul_4:product:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2&
$model_2/lstm_1/while/lstm_cell_1/add¼
(model_2/lstm_1/while/lstm_cell_1/SigmoidSigmoid(model_2/lstm_1/while/lstm_cell_1/add:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2*
(model_2/lstm_1/while/lstm_cell_1/Sigmoidã
1model_2/lstm_1/while/lstm_cell_1/ReadVariableOp_1ReadVariableOp:model_2_lstm_1_while_lstm_cell_1_readvariableop_resource_0* 
_output_shapes
:
*
dtype023
1model_2/lstm_1/while/lstm_cell_1/ReadVariableOp_1Á
6model_2/lstm_1/while/lstm_cell_1/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB"       28
6model_2/lstm_1/while/lstm_cell_1/strided_slice_1/stackÅ
8model_2/lstm_1/while/lstm_cell_1/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       2:
8model_2/lstm_1/while/lstm_cell_1/strided_slice_1/stack_1Å
8model_2/lstm_1/while/lstm_cell_1/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2:
8model_2/lstm_1/while/lstm_cell_1/strided_slice_1/stack_2Ð
0model_2/lstm_1/while/lstm_cell_1/strided_slice_1StridedSlice9model_2/lstm_1/while/lstm_cell_1/ReadVariableOp_1:value:0?model_2/lstm_1/while/lstm_cell_1/strided_slice_1/stack:output:0Amodel_2/lstm_1/while/lstm_cell_1/strided_slice_1/stack_1:output:0Amodel_2/lstm_1/while/lstm_cell_1/strided_slice_1/stack_2:output:0*
Index0*
T0* 
_output_shapes
:
*

begin_mask*
end_mask22
0model_2/lstm_1/while/lstm_cell_1/strided_slice_1ú
)model_2/lstm_1/while/lstm_cell_1/MatMul_5MatMul*model_2/lstm_1/while/lstm_cell_1/mul_1:z:09model_2/lstm_1/while/lstm_cell_1/strided_slice_1:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2+
)model_2/lstm_1/while/lstm_cell_1/MatMul_5ö
&model_2/lstm_1/while/lstm_cell_1/add_1AddV23model_2/lstm_1/while/lstm_cell_1/BiasAdd_1:output:03model_2/lstm_1/while/lstm_cell_1/MatMul_5:product:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2(
&model_2/lstm_1/while/lstm_cell_1/add_1Â
*model_2/lstm_1/while/lstm_cell_1/Sigmoid_1Sigmoid*model_2/lstm_1/while/lstm_cell_1/add_1:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2,
*model_2/lstm_1/while/lstm_cell_1/Sigmoid_1Þ
&model_2/lstm_1/while/lstm_cell_1/mul_4Mul.model_2/lstm_1/while/lstm_cell_1/Sigmoid_1:y:0"model_2_lstm_1_while_placeholder_4*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2(
&model_2/lstm_1/while/lstm_cell_1/mul_4ã
1model_2/lstm_1/while/lstm_cell_1/ReadVariableOp_2ReadVariableOp:model_2_lstm_1_while_lstm_cell_1_readvariableop_resource_0* 
_output_shapes
:
*
dtype023
1model_2/lstm_1/while/lstm_cell_1/ReadVariableOp_2Á
6model_2/lstm_1/while/lstm_cell_1/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB"       28
6model_2/lstm_1/while/lstm_cell_1/strided_slice_2/stackÅ
8model_2/lstm_1/while/lstm_cell_1/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       2:
8model_2/lstm_1/while/lstm_cell_1/strided_slice_2/stack_1Å
8model_2/lstm_1/while/lstm_cell_1/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2:
8model_2/lstm_1/while/lstm_cell_1/strided_slice_2/stack_2Ð
0model_2/lstm_1/while/lstm_cell_1/strided_slice_2StridedSlice9model_2/lstm_1/while/lstm_cell_1/ReadVariableOp_2:value:0?model_2/lstm_1/while/lstm_cell_1/strided_slice_2/stack:output:0Amodel_2/lstm_1/while/lstm_cell_1/strided_slice_2/stack_1:output:0Amodel_2/lstm_1/while/lstm_cell_1/strided_slice_2/stack_2:output:0*
Index0*
T0* 
_output_shapes
:
*

begin_mask*
end_mask22
0model_2/lstm_1/while/lstm_cell_1/strided_slice_2ú
)model_2/lstm_1/while/lstm_cell_1/MatMul_6MatMul*model_2/lstm_1/while/lstm_cell_1/mul_2:z:09model_2/lstm_1/while/lstm_cell_1/strided_slice_2:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2+
)model_2/lstm_1/while/lstm_cell_1/MatMul_6ö
&model_2/lstm_1/while/lstm_cell_1/add_2AddV23model_2/lstm_1/while/lstm_cell_1/BiasAdd_2:output:03model_2/lstm_1/while/lstm_cell_1/MatMul_6:product:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2(
&model_2/lstm_1/while/lstm_cell_1/add_2µ
%model_2/lstm_1/while/lstm_cell_1/TanhTanh*model_2/lstm_1/while/lstm_cell_1/add_2:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2'
%model_2/lstm_1/while/lstm_cell_1/Tanhã
&model_2/lstm_1/while/lstm_cell_1/mul_5Mul,model_2/lstm_1/while/lstm_cell_1/Sigmoid:y:0)model_2/lstm_1/while/lstm_cell_1/Tanh:y:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2(
&model_2/lstm_1/while/lstm_cell_1/mul_5ä
&model_2/lstm_1/while/lstm_cell_1/add_3AddV2*model_2/lstm_1/while/lstm_cell_1/mul_4:z:0*model_2/lstm_1/while/lstm_cell_1/mul_5:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2(
&model_2/lstm_1/while/lstm_cell_1/add_3ã
1model_2/lstm_1/while/lstm_cell_1/ReadVariableOp_3ReadVariableOp:model_2_lstm_1_while_lstm_cell_1_readvariableop_resource_0* 
_output_shapes
:
*
dtype023
1model_2/lstm_1/while/lstm_cell_1/ReadVariableOp_3Á
6model_2/lstm_1/while/lstm_cell_1/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB"       28
6model_2/lstm_1/while/lstm_cell_1/strided_slice_3/stackÅ
8model_2/lstm_1/while/lstm_cell_1/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        2:
8model_2/lstm_1/while/lstm_cell_1/strided_slice_3/stack_1Å
8model_2/lstm_1/while/lstm_cell_1/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2:
8model_2/lstm_1/while/lstm_cell_1/strided_slice_3/stack_2Ð
0model_2/lstm_1/while/lstm_cell_1/strided_slice_3StridedSlice9model_2/lstm_1/while/lstm_cell_1/ReadVariableOp_3:value:0?model_2/lstm_1/while/lstm_cell_1/strided_slice_3/stack:output:0Amodel_2/lstm_1/while/lstm_cell_1/strided_slice_3/stack_1:output:0Amodel_2/lstm_1/while/lstm_cell_1/strided_slice_3/stack_2:output:0*
Index0*
T0* 
_output_shapes
:
*

begin_mask*
end_mask22
0model_2/lstm_1/while/lstm_cell_1/strided_slice_3ú
)model_2/lstm_1/while/lstm_cell_1/MatMul_7MatMul*model_2/lstm_1/while/lstm_cell_1/mul_3:z:09model_2/lstm_1/while/lstm_cell_1/strided_slice_3:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2+
)model_2/lstm_1/while/lstm_cell_1/MatMul_7ö
&model_2/lstm_1/while/lstm_cell_1/add_4AddV23model_2/lstm_1/while/lstm_cell_1/BiasAdd_3:output:03model_2/lstm_1/while/lstm_cell_1/MatMul_7:product:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2(
&model_2/lstm_1/while/lstm_cell_1/add_4Â
*model_2/lstm_1/while/lstm_cell_1/Sigmoid_2Sigmoid*model_2/lstm_1/while/lstm_cell_1/add_4:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2,
*model_2/lstm_1/while/lstm_cell_1/Sigmoid_2¹
'model_2/lstm_1/while/lstm_cell_1/Tanh_1Tanh*model_2/lstm_1/while/lstm_cell_1/add_3:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2)
'model_2/lstm_1/while/lstm_cell_1/Tanh_1ç
&model_2/lstm_1/while/lstm_cell_1/mul_6Mul.model_2/lstm_1/while/lstm_cell_1/Sigmoid_2:y:0+model_2/lstm_1/while/lstm_cell_1/Tanh_1:y:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2(
&model_2/lstm_1/while/lstm_cell_1/mul_6
#model_2/lstm_1/while/Tile/multiplesConst*
_output_shapes
:*
dtype0*
valueB"      2%
#model_2/lstm_1/while/Tile/multiplesá
model_2/lstm_1/while/TileTileAmodel_2/lstm_1/while/TensorArrayV2Read_1/TensorListGetItem:item:0,model_2/lstm_1/while/Tile/multiples:output:0*
T0
*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
model_2/lstm_1/while/Tileñ
model_2/lstm_1/while/SelectV2SelectV2"model_2/lstm_1/while/Tile:output:0*model_2/lstm_1/while/lstm_cell_1/mul_6:z:0"model_2_lstm_1_while_placeholder_2*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
model_2/lstm_1/while/SelectV2
%model_2/lstm_1/while/Tile_1/multiplesConst*
_output_shapes
:*
dtype0*
valueB"      2'
%model_2/lstm_1/while/Tile_1/multiplesç
model_2/lstm_1/while/Tile_1TileAmodel_2/lstm_1/while/TensorArrayV2Read_1/TensorListGetItem:item:0.model_2/lstm_1/while/Tile_1/multiples:output:0*
T0
*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
model_2/lstm_1/while/Tile_1
%model_2/lstm_1/while/Tile_2/multiplesConst*
_output_shapes
:*
dtype0*
valueB"      2'
%model_2/lstm_1/while/Tile_2/multiplesç
model_2/lstm_1/while/Tile_2TileAmodel_2/lstm_1/while/TensorArrayV2Read_1/TensorListGetItem:item:0.model_2/lstm_1/while/Tile_2/multiples:output:0*
T0
*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
model_2/lstm_1/while/Tile_2÷
model_2/lstm_1/while/SelectV2_1SelectV2$model_2/lstm_1/while/Tile_1:output:0*model_2/lstm_1/while/lstm_cell_1/mul_6:z:0"model_2_lstm_1_while_placeholder_3*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2!
model_2/lstm_1/while/SelectV2_1÷
model_2/lstm_1/while/SelectV2_2SelectV2$model_2/lstm_1/while/Tile_2:output:0*model_2/lstm_1/while/lstm_cell_1/add_3:z:0"model_2_lstm_1_while_placeholder_4*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2!
model_2/lstm_1/while/SelectV2_2¦
9model_2/lstm_1/while/TensorArrayV2Write/TensorListSetItemTensorListSetItem"model_2_lstm_1_while_placeholder_1 model_2_lstm_1_while_placeholder&model_2/lstm_1/while/SelectV2:output:0*
_output_shapes
: *
element_dtype02;
9model_2/lstm_1/while/TensorArrayV2Write/TensorListSetItemz
model_2/lstm_1/while/add/yConst*
_output_shapes
: *
dtype0*
value	B :2
model_2/lstm_1/while/add/y¥
model_2/lstm_1/while/addAddV2 model_2_lstm_1_while_placeholder#model_2/lstm_1/while/add/y:output:0*
T0*
_output_shapes
: 2
model_2/lstm_1/while/add~
model_2/lstm_1/while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :2
model_2/lstm_1/while/add_1/yÁ
model_2/lstm_1/while/add_1AddV26model_2_lstm_1_while_model_2_lstm_1_while_loop_counter%model_2/lstm_1/while/add_1/y:output:0*
T0*
_output_shapes
: 2
model_2/lstm_1/while/add_1Ë
model_2/lstm_1/while/IdentityIdentitymodel_2/lstm_1/while/add_1:z:00^model_2/lstm_1/while/lstm_cell_1/ReadVariableOp2^model_2/lstm_1/while/lstm_cell_1/ReadVariableOp_12^model_2/lstm_1/while/lstm_cell_1/ReadVariableOp_22^model_2/lstm_1/while/lstm_cell_1/ReadVariableOp_36^model_2/lstm_1/while/lstm_cell_1/split/ReadVariableOp8^model_2/lstm_1/while/lstm_cell_1/split_1/ReadVariableOp*
T0*
_output_shapes
: 2
model_2/lstm_1/while/Identityí
model_2/lstm_1/while/Identity_1Identity<model_2_lstm_1_while_model_2_lstm_1_while_maximum_iterations0^model_2/lstm_1/while/lstm_cell_1/ReadVariableOp2^model_2/lstm_1/while/lstm_cell_1/ReadVariableOp_12^model_2/lstm_1/while/lstm_cell_1/ReadVariableOp_22^model_2/lstm_1/while/lstm_cell_1/ReadVariableOp_36^model_2/lstm_1/while/lstm_cell_1/split/ReadVariableOp8^model_2/lstm_1/while/lstm_cell_1/split_1/ReadVariableOp*
T0*
_output_shapes
: 2!
model_2/lstm_1/while/Identity_1Í
model_2/lstm_1/while/Identity_2Identitymodel_2/lstm_1/while/add:z:00^model_2/lstm_1/while/lstm_cell_1/ReadVariableOp2^model_2/lstm_1/while/lstm_cell_1/ReadVariableOp_12^model_2/lstm_1/while/lstm_cell_1/ReadVariableOp_22^model_2/lstm_1/while/lstm_cell_1/ReadVariableOp_36^model_2/lstm_1/while/lstm_cell_1/split/ReadVariableOp8^model_2/lstm_1/while/lstm_cell_1/split_1/ReadVariableOp*
T0*
_output_shapes
: 2!
model_2/lstm_1/while/Identity_2ú
model_2/lstm_1/while/Identity_3IdentityImodel_2/lstm_1/while/TensorArrayV2Write/TensorListSetItem:output_handle:00^model_2/lstm_1/while/lstm_cell_1/ReadVariableOp2^model_2/lstm_1/while/lstm_cell_1/ReadVariableOp_12^model_2/lstm_1/while/lstm_cell_1/ReadVariableOp_22^model_2/lstm_1/while/lstm_cell_1/ReadVariableOp_36^model_2/lstm_1/while/lstm_cell_1/split/ReadVariableOp8^model_2/lstm_1/while/lstm_cell_1/split_1/ReadVariableOp*
T0*
_output_shapes
: 2!
model_2/lstm_1/while/Identity_3é
model_2/lstm_1/while/Identity_4Identity&model_2/lstm_1/while/SelectV2:output:00^model_2/lstm_1/while/lstm_cell_1/ReadVariableOp2^model_2/lstm_1/while/lstm_cell_1/ReadVariableOp_12^model_2/lstm_1/while/lstm_cell_1/ReadVariableOp_22^model_2/lstm_1/while/lstm_cell_1/ReadVariableOp_36^model_2/lstm_1/while/lstm_cell_1/split/ReadVariableOp8^model_2/lstm_1/while/lstm_cell_1/split_1/ReadVariableOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2!
model_2/lstm_1/while/Identity_4ë
model_2/lstm_1/while/Identity_5Identity(model_2/lstm_1/while/SelectV2_1:output:00^model_2/lstm_1/while/lstm_cell_1/ReadVariableOp2^model_2/lstm_1/while/lstm_cell_1/ReadVariableOp_12^model_2/lstm_1/while/lstm_cell_1/ReadVariableOp_22^model_2/lstm_1/while/lstm_cell_1/ReadVariableOp_36^model_2/lstm_1/while/lstm_cell_1/split/ReadVariableOp8^model_2/lstm_1/while/lstm_cell_1/split_1/ReadVariableOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2!
model_2/lstm_1/while/Identity_5ë
model_2/lstm_1/while/Identity_6Identity(model_2/lstm_1/while/SelectV2_2:output:00^model_2/lstm_1/while/lstm_cell_1/ReadVariableOp2^model_2/lstm_1/while/lstm_cell_1/ReadVariableOp_12^model_2/lstm_1/while/lstm_cell_1/ReadVariableOp_22^model_2/lstm_1/while/lstm_cell_1/ReadVariableOp_36^model_2/lstm_1/while/lstm_cell_1/split/ReadVariableOp8^model_2/lstm_1/while/lstm_cell_1/split_1/ReadVariableOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2!
model_2/lstm_1/while/Identity_6"G
model_2_lstm_1_while_identity&model_2/lstm_1/while/Identity:output:0"K
model_2_lstm_1_while_identity_1(model_2/lstm_1/while/Identity_1:output:0"K
model_2_lstm_1_while_identity_2(model_2/lstm_1/while/Identity_2:output:0"K
model_2_lstm_1_while_identity_3(model_2/lstm_1/while/Identity_3:output:0"K
model_2_lstm_1_while_identity_4(model_2/lstm_1/while/Identity_4:output:0"K
model_2_lstm_1_while_identity_5(model_2/lstm_1/while/Identity_5:output:0"K
model_2_lstm_1_while_identity_6(model_2/lstm_1/while/Identity_6:output:0"v
8model_2_lstm_1_while_lstm_cell_1_readvariableop_resource:model_2_lstm_1_while_lstm_cell_1_readvariableop_resource_0"
@model_2_lstm_1_while_lstm_cell_1_split_1_readvariableop_resourceBmodel_2_lstm_1_while_lstm_cell_1_split_1_readvariableop_resource_0"
>model_2_lstm_1_while_lstm_cell_1_split_readvariableop_resource@model_2_lstm_1_while_lstm_cell_1_split_readvariableop_resource_0"h
1model_2_lstm_1_while_model_2_lstm_1_strided_slice3model_2_lstm_1_while_model_2_lstm_1_strided_slice_0"ì
smodel_2_lstm_1_while_tensorarrayv2read_1_tensorlistgetitem_model_2_lstm_1_tensorarrayunstack_1_tensorlistfromtensorumodel_2_lstm_1_while_tensorarrayv2read_1_tensorlistgetitem_model_2_lstm_1_tensorarrayunstack_1_tensorlistfromtensor_0"ä
omodel_2_lstm_1_while_tensorarrayv2read_tensorlistgetitem_model_2_lstm_1_tensorarrayunstack_tensorlistfromtensorqmodel_2_lstm_1_while_tensorarrayv2read_tensorlistgetitem_model_2_lstm_1_tensorarrayunstack_tensorlistfromtensor_0*i
_input_shapesX
V: : : : :ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ: : : :::2b
/model_2/lstm_1/while/lstm_cell_1/ReadVariableOp/model_2/lstm_1/while/lstm_cell_1/ReadVariableOp2f
1model_2/lstm_1/while/lstm_cell_1/ReadVariableOp_11model_2/lstm_1/while/lstm_cell_1/ReadVariableOp_12f
1model_2/lstm_1/while/lstm_cell_1/ReadVariableOp_21model_2/lstm_1/while/lstm_cell_1/ReadVariableOp_22f
1model_2/lstm_1/while/lstm_cell_1/ReadVariableOp_31model_2/lstm_1/while/lstm_cell_1/ReadVariableOp_32n
5model_2/lstm_1/while/lstm_cell_1/split/ReadVariableOp5model_2/lstm_1/while/lstm_cell_1/split/ReadVariableOp2r
7model_2/lstm_1/while/lstm_cell_1/split_1/ReadVariableOp7model_2/lstm_1/while/lstm_cell_1/split_1/ReadVariableOp: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :.*
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ:.*
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ:.*
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ:

_output_shapes
: :

_output_shapes
: :	

_output_shapes
: 
Ü	

while_cond_14901
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_placeholder_4
while_less_strided_slice3
/while_while_cond_14901___redundant_placeholder03
/while_while_cond_14901___redundant_placeholder13
/while_while_cond_14901___redundant_placeholder23
/while_while_cond_14901___redundant_placeholder33
/while_while_cond_14901___redundant_placeholder4
while_identity
n

while/LessLesswhile_placeholderwhile_less_strided_slice*
T0*
_output_shapes
: 2

while/Less]
while/IdentityIdentitywhile/Less:z:0*
T0
*
_output_shapes
: 2
while/Identity")
while_identitywhile/Identity:output:0*m
_input_shapes\
Z: : : : :ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ: :::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :.*
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ:.*
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ:.*
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ:

_output_shapes
: :

_output_shapes
::	

_output_shapes
:
Ð
H
,__inference_activation_1_layer_call_fn_16044

inputs
identityÓ
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *5
_output_shapes#
!:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *P
fKRI
G__inference_activation_1_layer_call_and_return_conditional_losses_144462
PartitionedCallz
IdentityIdentityPartitionedCall:output:0*
T0*5
_output_shapes#
!:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*4
_input_shapes#
!:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ:] Y
5
_output_shapes#
!:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
ö$

while_body_14189
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0
while_lstm_cell_1_14213_0
while_lstm_cell_1_14215_0
while_lstm_cell_1_14217_0
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_identity_5
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor
while_lstm_cell_1_14213
while_lstm_cell_1_14215
while_lstm_cell_1_14217¢)while/lstm_cell_1/StatefulPartitionedCallÃ
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ   29
7while/TensorArrayV2Read/TensorListGetItem/element_shapeÔ
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
element_dtype02+
)while/TensorArrayV2Read/TensorListGetItemÚ
)while/lstm_cell_1/StatefulPartitionedCallStatefulPartitionedCall0while/TensorArrayV2Read/TensorListGetItem:item:0while_placeholder_2while_placeholder_3while_lstm_cell_1_14213_0while_lstm_cell_1_14215_0while_lstm_cell_1_14217_0*
Tin

2*
Tout
2*
_collective_manager_ids
 *P
_output_shapes>
<:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *O
fJRH
F__inference_lstm_cell_1_layer_call_and_return_conditional_losses_138142+
)while/lstm_cell_1/StatefulPartitionedCallö
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholder2while/lstm_cell_1/StatefulPartitionedCall:output:0*
_output_shapes
: *
element_dtype02,
*while/TensorArrayV2Write/TensorListSetItem\
while/add/yConst*
_output_shapes
: *
dtype0*
value	B :2
while/add/yi
	while/addAddV2while_placeholderwhile/add/y:output:0*
T0*
_output_shapes
: 2
	while/add`
while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :2
while/add_1/yv
while/add_1AddV2while_while_loop_counterwhile/add_1/y:output:0*
T0*
_output_shapes
: 2
while/add_1
while/IdentityIdentitywhile/add_1:z:0*^while/lstm_cell_1/StatefulPartitionedCall*
T0*
_output_shapes
: 2
while/Identity
while/Identity_1Identitywhile_while_maximum_iterations*^while/lstm_cell_1/StatefulPartitionedCall*
T0*
_output_shapes
: 2
while/Identity_1
while/Identity_2Identitywhile/add:z:0*^while/lstm_cell_1/StatefulPartitionedCall*
T0*
_output_shapes
: 2
while/Identity_2¹
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0*^while/lstm_cell_1/StatefulPartitionedCall*
T0*
_output_shapes
: 2
while/Identity_3Ã
while/Identity_4Identity2while/lstm_cell_1/StatefulPartitionedCall:output:1*^while/lstm_cell_1/StatefulPartitionedCall*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
while/Identity_4Ã
while/Identity_5Identity2while/lstm_cell_1/StatefulPartitionedCall:output:2*^while/lstm_cell_1/StatefulPartitionedCall*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
while/Identity_5")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"-
while_identity_5while/Identity_5:output:0"4
while_lstm_cell_1_14213while_lstm_cell_1_14213_0"4
while_lstm_cell_1_14215while_lstm_cell_1_14215_0"4
while_lstm_cell_1_14217while_lstm_cell_1_14217_0"0
while_strided_slice_1while_strided_slice_1_0"¨
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*S
_input_shapesB
@: : : : :ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ: : :::2V
)while/lstm_cell_1/StatefulPartitionedCall)while/lstm_cell_1/StatefulPartitionedCall: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :.*
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ:.*
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ:

_output_shapes
: :

_output_shapes
: 
É
ö
'__inference_model_2_layer_call_fn_16017
inputs_0
inputs_1
inputs_2
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
	unknown_4
identity

identity_1

identity_2¢StatefulPartitionedCallõ
StatefulPartitionedCallStatefulPartitionedCallinputs_0inputs_1inputs_2unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4*
Tin
2	*
Tout
2*
_collective_manager_ids
 *\
_output_shapesJ
H:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ7:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ*(
_read_only_resource_inputs

*-
config_proto

CPU

GPU 2J 8 *K
fFRD
B__inference_model_2_layer_call_and_return_conditional_losses_152682
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ72

Identity

Identity_1Identity StatefulPartitionedCall:output:1^StatefulPartitionedCall*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity_1

Identity_2Identity StatefulPartitionedCall:output:2^StatefulPartitionedCall*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity_2"
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0*o
_input_shapes^
\:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ::::::22
StatefulPartitionedCallStatefulPartitionedCall:Z V
0
_output_shapes
:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
"
_user_specified_name
inputs/0:RN
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
"
_user_specified_name
inputs/1:RN
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
"
_user_specified_name
inputs/2
É
ö
'__inference_model_2_layer_call_fn_15994
inputs_0
inputs_1
inputs_2
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
	unknown_4
identity

identity_1

identity_2¢StatefulPartitionedCallõ
StatefulPartitionedCallStatefulPartitionedCallinputs_0inputs_1inputs_2unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4*
Tin
2	*
Tout
2*
_collective_manager_ids
 *\
_output_shapesJ
H:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ7:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ*(
_read_only_resource_inputs

*-
config_proto

CPU

GPU 2J 8 *K
fFRD
B__inference_model_2_layer_call_and_return_conditional_losses_152162
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ72

Identity

Identity_1Identity StatefulPartitionedCall:output:1^StatefulPartitionedCall*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity_1

Identity_2Identity StatefulPartitionedCall:output:2^StatefulPartitionedCall*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity_2"
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0*o
_input_shapes^
\:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ::::::22
StatefulPartitionedCallStatefulPartitionedCall:Z V
0
_output_shapes
:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
"
_user_specified_name
inputs/0:RN
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
"
_user_specified_name
inputs/1:RN
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
"
_user_specified_name
inputs/2
ý 
ó
B__inference_model_2_layer_call_and_return_conditional_losses_15268

inputs
inputs_1
inputs_2
embedding_1_15244
lstm_1_15251
lstm_1_15253
lstm_1_15255
dense_15260
dense_15262
identity

identity_1

identity_2¢dense/StatefulPartitionedCall¢#embedding_1/StatefulPartitionedCall¢lstm_1/StatefulPartitionedCall
#embedding_1/StatefulPartitionedCallStatefulPartitionedCallinputsembedding_1_15244*
Tin
2*
Tout
2*
_collective_manager_ids
 *5
_output_shapes#
!:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ*#
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *O
fJRH
F__inference_embedding_1_layer_call_and_return_conditional_losses_144272%
#embedding_1/StatefulPartitionedCallu
embedding_1/NotEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *    2
embedding_1/NotEqual/y
embedding_1/NotEqualNotEqualinputsembedding_1/NotEqual/y:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2
embedding_1/NotEqual
activation_1/PartitionedCallPartitionedCall,embedding_1/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *5
_output_shapes#
!:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *P
fKRI
G__inference_activation_1_layer_call_and_return_conditional_losses_144462
activation_1/PartitionedCall
dropout_1/PartitionedCallPartitionedCall%activation_1/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *5
_output_shapes#
!:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *M
fHRF
D__inference_dropout_1_layer_call_and_return_conditional_losses_144712
dropout_1/PartitionedCall
lstm_1/StatefulPartitionedCallStatefulPartitionedCall"dropout_1/PartitionedCall:output:0embedding_1/NotEqual:z:0inputs_1inputs_2lstm_1_15251lstm_1_15253lstm_1_15255*
Tin
	2
*
Tout
2*
_collective_manager_ids
 *]
_output_shapesK
I:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *J
fERC
A__inference_lstm_1_layer_call_and_return_conditional_losses_150512 
lstm_1/StatefulPartitionedCall°
dense/StatefulPartitionedCallStatefulPartitionedCall'lstm_1/StatefulPartitionedCall:output:0dense_15260dense_15262*
Tin
2*
Tout
2*
_collective_manager_ids
 *4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ7*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *I
fDRB
@__inference_dense_layer_call_and_return_conditional_losses_151342
dense/StatefulPartitionedCallî
IdentityIdentity&dense/StatefulPartitionedCall:output:0^dense/StatefulPartitionedCall$^embedding_1/StatefulPartitionedCall^lstm_1/StatefulPartitionedCall*
T0*4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ72

Identityç

Identity_1Identity'lstm_1/StatefulPartitionedCall:output:1^dense/StatefulPartitionedCall$^embedding_1/StatefulPartitionedCall^lstm_1/StatefulPartitionedCall*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity_1ç

Identity_2Identity'lstm_1/StatefulPartitionedCall:output:2^dense/StatefulPartitionedCall$^embedding_1/StatefulPartitionedCall^lstm_1/StatefulPartitionedCall*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity_2"
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0*o
_input_shapes^
\:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ::::::2>
dense/StatefulPartitionedCalldense/StatefulPartitionedCall2J
#embedding_1/StatefulPartitionedCall#embedding_1/StatefulPartitionedCall2@
lstm_1/StatefulPartitionedCalllstm_1/StatefulPartitionedCall:X T
0
_output_shapes
:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs:PL
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs:PL
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs



F__inference_embedding_1_layer_call_and_return_conditional_losses_14427

inputs
embedding_lookup_14421
identity¢embedding_lookupf
CastCastinputs*

DstT0*

SrcT0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2
Cast
embedding_lookupResourceGatherembedding_lookup_14421Cast:y:0",/job:localhost/replica:0/task:0/device:CPU:0*
Tindices0*)
_class
loc:@embedding_lookup/14421*5
_output_shapes#
!:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ*
dtype02
embedding_lookupö
embedding_lookup/IdentityIdentityembedding_lookup:output:0",/job:localhost/replica:0/task:0/device:CPU:0*
T0*)
_class
loc:@embedding_lookup/14421*5
_output_shapes#
!:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2
embedding_lookup/Identityª
embedding_lookup/Identity_1Identity"embedding_lookup/Identity:output:0*
T0*5
_output_shapes#
!:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2
embedding_lookup/Identity_1
IdentityIdentity$embedding_lookup/Identity_1:output:0^embedding_lookup*
T0*5
_output_shapes#
!:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*3
_input_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ:2$
embedding_lookupembedding_lookup:X T
0
_output_shapes
:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
í
c
G__inference_activation_1_layer_call_and_return_conditional_losses_16039

inputs
identity\
ReluReluinputs*
T0*5
_output_shapes#
!:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2
Relut
IdentityIdentityRelu:activations:0*
T0*5
_output_shapes#
!:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*4
_input_shapes#
!:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ:] Y
5
_output_shapes#
!:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
À
ó
'__inference_model_2_layer_call_fn_15235
input_2
input_4
input_3
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
	unknown_4
identity

identity_1

identity_2¢StatefulPartitionedCallò
StatefulPartitionedCallStatefulPartitionedCallinput_2input_4input_3unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4*
Tin
2	*
Tout
2*
_collective_manager_ids
 *\
_output_shapesJ
H:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ7:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ*(
_read_only_resource_inputs

*-
config_proto

CPU

GPU 2J 8 *K
fFRD
B__inference_model_2_layer_call_and_return_conditional_losses_152162
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ72

Identity

Identity_1Identity StatefulPartitionedCall:output:1^StatefulPartitionedCall*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity_1

Identity_2Identity StatefulPartitionedCall:output:2^StatefulPartitionedCall*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity_2"
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0*o
_input_shapes^
\:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ::::::22
StatefulPartitionedCallStatefulPartitionedCall:Y U
0
_output_shapes
:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
!
_user_specified_name	input_2:QM
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
!
_user_specified_name	input_4:QM
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
!
_user_specified_name	input_3

z
%__inference_dense_layer_call_fn_17301

inputs
unknown
	unknown_0
identity¢StatefulPartitionedCallý
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ7*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *I
fDRB
@__inference_dense_layer_call_and_return_conditional_losses_151342
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ72

Identity"
identityIdentity:output:0*<
_input_shapes+
):ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ::22
StatefulPartitionedCallStatefulPartitionedCall:] Y
5
_output_shapes#
!:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
úH

F__inference_lstm_cell_1_layer_call_and_return_conditional_losses_17487

inputs
states_0
states_1!
split_readvariableop_resource#
split_1_readvariableop_resource
readvariableop_resource
identity

identity_1

identity_2¢ReadVariableOp¢ReadVariableOp_1¢ReadVariableOp_2¢ReadVariableOp_3¢split/ReadVariableOp¢split_1/ReadVariableOpZ
ones_like/ShapeShapestates_0*
T0*
_output_shapes
:2
ones_like/Shapeg
ones_like/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ?2
ones_like/Const
	ones_likeFillones_like/Shape:output:0ones_like/Const:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
	ones_likeP
ConstConst*
_output_shapes
: *
dtype0*
value	B :2
Constd
split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2
split/split_dim
split/ReadVariableOpReadVariableOpsplit_readvariableop_resource* 
_output_shapes
:
*
dtype02
split/ReadVariableOp¯
splitSplitsplit/split_dim:output:0split/ReadVariableOp:value:0*
T0*D
_output_shapes2
0:
:
:
:
*
	num_split2
splite
MatMulMatMulinputssplit:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
MatMuli
MatMul_1MatMulinputssplit:output:1*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

MatMul_1i
MatMul_2MatMulinputssplit:output:2*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

MatMul_2i
MatMul_3MatMulinputssplit:output:3*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

MatMul_3T
Const_1Const*
_output_shapes
: *
dtype0*
value	B :2	
Const_1h
split_1/split_dimConst*
_output_shapes
: *
dtype0*
value	B : 2
split_1/split_dim
split_1/ReadVariableOpReadVariableOpsplit_1_readvariableop_resource*
_output_shapes	
:*
dtype02
split_1/ReadVariableOp£
split_1Splitsplit_1/split_dim:output:0split_1/ReadVariableOp:value:0*
T0*0
_output_shapes
::::*
	num_split2	
split_1t
BiasAddBiasAddMatMul:product:0split_1:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2	
BiasAddz
	BiasAdd_1BiasAddMatMul_1:product:0split_1:output:1*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
	BiasAdd_1z
	BiasAdd_2BiasAddMatMul_2:product:0split_1:output:2*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
	BiasAdd_2z
	BiasAdd_3BiasAddMatMul_3:product:0split_1:output:3*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
	BiasAdd_3b
mulMulstates_0ones_like:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
mulf
mul_1Mulstates_0ones_like:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
mul_1f
mul_2Mulstates_0ones_like:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
mul_2f
mul_3Mulstates_0ones_like:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
mul_3z
ReadVariableOpReadVariableOpreadvariableop_resource* 
_output_shapes
:
*
dtype02
ReadVariableOp{
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB"        2
strided_slice/stack
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       2
strided_slice/stack_1
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2
strided_slice/stack_2þ
strided_sliceStridedSliceReadVariableOp:value:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0* 
_output_shapes
:
*

begin_mask*
end_mask2
strided_slicer
MatMul_4MatMulmul:z:0strided_slice:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

MatMul_4l
addAddV2BiasAdd:output:0MatMul_4:product:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
addY
SigmoidSigmoidadd:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2	
Sigmoid~
ReadVariableOp_1ReadVariableOpreadvariableop_resource* 
_output_shapes
:
*
dtype02
ReadVariableOp_1
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB"       2
strided_slice_1/stack
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       2
strided_slice_1/stack_1
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2
strided_slice_1/stack_2
strided_slice_1StridedSliceReadVariableOp_1:value:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0* 
_output_shapes
:
*

begin_mask*
end_mask2
strided_slice_1v
MatMul_5MatMul	mul_1:z:0strided_slice_1:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

MatMul_5r
add_1AddV2BiasAdd_1:output:0MatMul_5:product:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
add_1_
	Sigmoid_1Sigmoid	add_1:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
	Sigmoid_1a
mul_4MulSigmoid_1:y:0states_1*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
mul_4~
ReadVariableOp_2ReadVariableOpreadvariableop_resource* 
_output_shapes
:
*
dtype02
ReadVariableOp_2
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB"       2
strided_slice_2/stack
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       2
strided_slice_2/stack_1
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2
strided_slice_2/stack_2
strided_slice_2StridedSliceReadVariableOp_2:value:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0* 
_output_shapes
:
*

begin_mask*
end_mask2
strided_slice_2v
MatMul_6MatMul	mul_2:z:0strided_slice_2:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

MatMul_6r
add_2AddV2BiasAdd_2:output:0MatMul_6:product:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
add_2R
TanhTanh	add_2:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
Tanh_
mul_5MulSigmoid:y:0Tanh:y:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
mul_5`
add_3AddV2	mul_4:z:0	mul_5:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
add_3~
ReadVariableOp_3ReadVariableOpreadvariableop_resource* 
_output_shapes
:
*
dtype02
ReadVariableOp_3
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB"       2
strided_slice_3/stack
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        2
strided_slice_3/stack_1
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2
strided_slice_3/stack_2
strided_slice_3StridedSliceReadVariableOp_3:value:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0* 
_output_shapes
:
*

begin_mask*
end_mask2
strided_slice_3v
MatMul_7MatMul	mul_3:z:0strided_slice_3:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

MatMul_7r
add_4AddV2BiasAdd_3:output:0MatMul_7:product:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
add_4_
	Sigmoid_2Sigmoid	add_4:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
	Sigmoid_2V
Tanh_1Tanh	add_3:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
Tanh_1c
mul_6MulSigmoid_2:y:0
Tanh_1:y:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
mul_6Ø
IdentityIdentity	mul_6:z:0^ReadVariableOp^ReadVariableOp_1^ReadVariableOp_2^ReadVariableOp_3^split/ReadVariableOp^split_1/ReadVariableOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

IdentityÜ

Identity_1Identity	mul_6:z:0^ReadVariableOp^ReadVariableOp_1^ReadVariableOp_2^ReadVariableOp_3^split/ReadVariableOp^split_1/ReadVariableOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity_1Ü

Identity_2Identity	add_3:z:0^ReadVariableOp^ReadVariableOp_1^ReadVariableOp_2^ReadVariableOp_3^split/ReadVariableOp^split_1/ReadVariableOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity_2"
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0*[
_input_shapesJ
H:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:::2 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_12$
ReadVariableOp_2ReadVariableOp_22$
ReadVariableOp_3ReadVariableOp_32,
split/ReadVariableOpsplit/ReadVariableOp20
split_1/ReadVariableOpsplit_1/ReadVariableOp:P L
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs:RN
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
"
_user_specified_name
states/0:RN
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
"
_user_specified_name
states/1
ÿ 
ò
B__inference_model_2_layer_call_and_return_conditional_losses_15182
input_2
input_4
input_3
embedding_1_15158
lstm_1_15165
lstm_1_15167
lstm_1_15169
dense_15174
dense_15176
identity

identity_1

identity_2¢dense/StatefulPartitionedCall¢#embedding_1/StatefulPartitionedCall¢lstm_1/StatefulPartitionedCall
#embedding_1/StatefulPartitionedCallStatefulPartitionedCallinput_2embedding_1_15158*
Tin
2*
Tout
2*
_collective_manager_ids
 *5
_output_shapes#
!:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ*#
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *O
fJRH
F__inference_embedding_1_layer_call_and_return_conditional_losses_144272%
#embedding_1/StatefulPartitionedCallu
embedding_1/NotEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *    2
embedding_1/NotEqual/y
embedding_1/NotEqualNotEqualinput_2embedding_1/NotEqual/y:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2
embedding_1/NotEqual
activation_1/PartitionedCallPartitionedCall,embedding_1/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *5
_output_shapes#
!:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *P
fKRI
G__inference_activation_1_layer_call_and_return_conditional_losses_144462
activation_1/PartitionedCall
dropout_1/PartitionedCallPartitionedCall%activation_1/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *5
_output_shapes#
!:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *M
fHRF
D__inference_dropout_1_layer_call_and_return_conditional_losses_144712
dropout_1/PartitionedCall
lstm_1/StatefulPartitionedCallStatefulPartitionedCall"dropout_1/PartitionedCall:output:0embedding_1/NotEqual:z:0input_4input_3lstm_1_15165lstm_1_15167lstm_1_15169*
Tin
	2
*
Tout
2*
_collective_manager_ids
 *]
_output_shapesK
I:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *J
fERC
A__inference_lstm_1_layer_call_and_return_conditional_losses_150512 
lstm_1/StatefulPartitionedCall°
dense/StatefulPartitionedCallStatefulPartitionedCall'lstm_1/StatefulPartitionedCall:output:0dense_15174dense_15176*
Tin
2*
Tout
2*
_collective_manager_ids
 *4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ7*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *I
fDRB
@__inference_dense_layer_call_and_return_conditional_losses_151342
dense/StatefulPartitionedCallî
IdentityIdentity&dense/StatefulPartitionedCall:output:0^dense/StatefulPartitionedCall$^embedding_1/StatefulPartitionedCall^lstm_1/StatefulPartitionedCall*
T0*4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ72

Identityç

Identity_1Identity'lstm_1/StatefulPartitionedCall:output:1^dense/StatefulPartitionedCall$^embedding_1/StatefulPartitionedCall^lstm_1/StatefulPartitionedCall*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity_1ç

Identity_2Identity'lstm_1/StatefulPartitionedCall:output:2^dense/StatefulPartitionedCall$^embedding_1/StatefulPartitionedCall^lstm_1/StatefulPartitionedCall*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity_2"
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0*o
_input_shapes^
\:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ::::::2>
dense/StatefulPartitionedCalldense/StatefulPartitionedCall2J
#embedding_1/StatefulPartitionedCall#embedding_1/StatefulPartitionedCall2@
lstm_1/StatefulPartitionedCalllstm_1/StatefulPartitionedCall:Y U
0
_output_shapes
:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
!
_user_specified_name	input_2:QM
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
!
_user_specified_name	input_4:QM
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
!
_user_specified_name	input_3"±L
saver_filename:0StatefulPartitionedCall_1:0StatefulPartitionedCall_28"
saved_model_main_op

NoOp*>
__saved_model_init_op%#
__saved_model_init_op

NoOp*¶
serving_default¢
D
input_29
serving_default_input_2:0ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
<
input_31
serving_default_input_3:0ÿÿÿÿÿÿÿÿÿ
<
input_41
serving_default_input_4:0ÿÿÿÿÿÿÿÿÿF
dense=
StatefulPartitionedCall:0ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ7;
lstm_11
StatefulPartitionedCall:1ÿÿÿÿÿÿÿÿÿ=
lstm_1_11
StatefulPartitionedCall:2ÿÿÿÿÿÿÿÿÿtensorflow/serving/predict:Û
è=
layer-0
layer_with_weights-0
layer-1
layer-2
layer-3
layer-4
layer-5
layer_with_weights-1
layer-6
layer_with_weights-2
layer-7
	regularization_losses

	variables
trainable_variables
	keras_api

signatures
R_default_save_signature
S__call__
*T&call_and_return_all_conditional_losses"ö:
_tf_keras_networkÚ:{"class_name": "Functional", "name": "model_2", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "must_restore_from_config": false, "config": {"name": "model_2", "layers": [{"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, null]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "input_2"}, "name": "input_2", "inbound_nodes": []}, {"class_name": "Embedding", "config": {"name": "embedding_1", "trainable": true, "batch_input_shape": {"class_name": "__tuple__", "items": [null, null]}, "dtype": "float32", "input_dim": 55, "output_dim": 256, "embeddings_initializer": {"class_name": "RandomUniform", "config": {"minval": -0.05, "maxval": 0.05, "seed": null}}, "embeddings_regularizer": null, "activity_regularizer": null, "embeddings_constraint": null, "mask_zero": true, "input_length": null}, "name": "embedding_1", "inbound_nodes": [[["input_2", 0, 0, {}]]]}, {"class_name": "Activation", "config": {"name": "activation_1", "trainable": true, "dtype": "float32", "activation": "relu"}, "name": "activation_1", "inbound_nodes": [[["embedding_1", 0, 0, {}]]]}, {"class_name": "Dropout", "config": {"name": "dropout_1", "trainable": true, "dtype": "float32", "rate": 0.3, "noise_shape": null, "seed": null}, "name": "dropout_1", "inbound_nodes": [[["activation_1", 0, 0, {}]]]}, {"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 256]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "input_4"}, "name": "input_4", "inbound_nodes": []}, {"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 256]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "input_3"}, "name": "input_3", "inbound_nodes": []}, {"class_name": "LSTM", "config": {"name": "lstm_1", "trainable": true, "dtype": "float32", "return_sequences": true, "return_state": true, "go_backwards": false, "stateful": false, "unroll": false, "time_major": false, "units": 256, "activation": "tanh", "recurrent_activation": "sigmoid", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "recurrent_initializer": {"class_name": "Orthogonal", "config": {"gain": 1.0, "seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "unit_forget_bias": true, "kernel_regularizer": null, "recurrent_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "recurrent_constraint": null, "bias_constraint": null, "dropout": 0.0, "recurrent_dropout": 0.1, "implementation": 1}, "name": "lstm_1", "inbound_nodes": [[["dropout_1", 0, 0, {}], ["input_4", 0, 0, {}], ["input_3", 0, 0, {}]]]}, {"class_name": "Dense", "config": {"name": "dense", "trainable": true, "dtype": "float32", "units": 55, "activation": "softmax", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "dense", "inbound_nodes": [[["lstm_1", 0, 0, {}]]]}], "input_layers": [["input_2", 0, 0], ["input_4", 0, 0], ["input_3", 0, 0]], "output_layers": [["dense", 0, 0], ["lstm_1", 0, 1], ["lstm_1", 0, 2]]}, "input_spec": [{"class_name": "InputSpec", "config": {"dtype": null, "shape": {"class_name": "__tuple__", "items": [null, null]}, "ndim": 2, "max_ndim": null, "min_ndim": null, "axes": {}}}, {"class_name": "InputSpec", "config": {"dtype": null, "shape": {"class_name": "__tuple__", "items": [null, 256]}, "ndim": 2, "max_ndim": null, "min_ndim": null, "axes": {}}}, {"class_name": "InputSpec", "config": {"dtype": null, "shape": {"class_name": "__tuple__", "items": [null, 256]}, "ndim": 2, "max_ndim": null, "min_ndim": null, "axes": {}}}], "build_input_shape": [{"class_name": "TensorShape", "items": [null, null]}, {"class_name": "TensorShape", "items": [null, 256]}, {"class_name": "TensorShape", "items": [null, 256]}], "is_graph_network": true, "keras_version": "2.4.0", "backend": "tensorflow", "model_config": {"class_name": "Functional", "config": {"name": "model_2", "layers": [{"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, null]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "input_2"}, "name": "input_2", "inbound_nodes": []}, {"class_name": "Embedding", "config": {"name": "embedding_1", "trainable": true, "batch_input_shape": {"class_name": "__tuple__", "items": [null, null]}, "dtype": "float32", "input_dim": 55, "output_dim": 256, "embeddings_initializer": {"class_name": "RandomUniform", "config": {"minval": -0.05, "maxval": 0.05, "seed": null}}, "embeddings_regularizer": null, "activity_regularizer": null, "embeddings_constraint": null, "mask_zero": true, "input_length": null}, "name": "embedding_1", "inbound_nodes": [[["input_2", 0, 0, {}]]]}, {"class_name": "Activation", "config": {"name": "activation_1", "trainable": true, "dtype": "float32", "activation": "relu"}, "name": "activation_1", "inbound_nodes": [[["embedding_1", 0, 0, {}]]]}, {"class_name": "Dropout", "config": {"name": "dropout_1", "trainable": true, "dtype": "float32", "rate": 0.3, "noise_shape": null, "seed": null}, "name": "dropout_1", "inbound_nodes": [[["activation_1", 0, 0, {}]]]}, {"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 256]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "input_4"}, "name": "input_4", "inbound_nodes": []}, {"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 256]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "input_3"}, "name": "input_3", "inbound_nodes": []}, {"class_name": "LSTM", "config": {"name": "lstm_1", "trainable": true, "dtype": "float32", "return_sequences": true, "return_state": true, "go_backwards": false, "stateful": false, "unroll": false, "time_major": false, "units": 256, "activation": "tanh", "recurrent_activation": "sigmoid", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "recurrent_initializer": {"class_name": "Orthogonal", "config": {"gain": 1.0, "seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "unit_forget_bias": true, "kernel_regularizer": null, "recurrent_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "recurrent_constraint": null, "bias_constraint": null, "dropout": 0.0, "recurrent_dropout": 0.1, "implementation": 1}, "name": "lstm_1", "inbound_nodes": [[["dropout_1", 0, 0, {}], ["input_4", 0, 0, {}], ["input_3", 0, 0, {}]]]}, {"class_name": "Dense", "config": {"name": "dense", "trainable": true, "dtype": "float32", "units": 55, "activation": "softmax", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "dense", "inbound_nodes": [[["lstm_1", 0, 0, {}]]]}], "input_layers": [["input_2", 0, 0], ["input_4", 0, 0], ["input_3", 0, 0]], "output_layers": [["dense", 0, 0], ["lstm_1", 0, 1], ["lstm_1", 0, 2]]}}}
ï"ì
_tf_keras_input_layerÌ{"class_name": "InputLayer", "name": "input_2", "dtype": "float32", "sparse": false, "ragged": false, "batch_input_shape": {"class_name": "__tuple__", "items": [null, null]}, "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, null]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "input_2"}}
®

embeddings
regularization_losses
	variables
trainable_variables
	keras_api
U__call__
*V&call_and_return_all_conditional_losses"
_tf_keras_layerõ{"class_name": "Embedding", "name": "embedding_1", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": {"class_name": "__tuple__", "items": [null, null]}, "stateful": false, "must_restore_from_config": false, "config": {"name": "embedding_1", "trainable": true, "batch_input_shape": {"class_name": "__tuple__", "items": [null, null]}, "dtype": "float32", "input_dim": 55, "output_dim": 256, "embeddings_initializer": {"class_name": "RandomUniform", "config": {"minval": -0.05, "maxval": 0.05, "seed": null}}, "embeddings_regularizer": null, "activity_regularizer": null, "embeddings_constraint": null, "mask_zero": true, "input_length": null}, "build_input_shape": {"class_name": "TensorShape", "items": [null, null]}}
Õ
regularization_losses
	variables
trainable_variables
	keras_api
W__call__
*X&call_and_return_all_conditional_losses"Æ
_tf_keras_layer¬{"class_name": "Activation", "name": "activation_1", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "activation_1", "trainable": true, "dtype": "float32", "activation": "relu"}}
å
regularization_losses
	variables
trainable_variables
	keras_api
Y__call__
*Z&call_and_return_all_conditional_losses"Ö
_tf_keras_layer¼{"class_name": "Dropout", "name": "dropout_1", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "dropout_1", "trainable": true, "dtype": "float32", "rate": 0.3, "noise_shape": null, "seed": null}}
í"ê
_tf_keras_input_layerÊ{"class_name": "InputLayer", "name": "input_4", "dtype": "float32", "sparse": false, "ragged": false, "batch_input_shape": {"class_name": "__tuple__", "items": [null, 256]}, "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 256]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "input_4"}}
í"ê
_tf_keras_input_layerÊ{"class_name": "InputLayer", "name": "input_3", "dtype": "float32", "sparse": false, "ragged": false, "batch_input_shape": {"class_name": "__tuple__", "items": [null, 256]}, "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 256]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "input_3"}}
¬
cell

state_spec
regularization_losses
	variables
trainable_variables
 	keras_api
[__call__
*\&call_and_return_all_conditional_losses"
_tf_keras_rnn_layerå
{"class_name": "LSTM", "name": "lstm_1", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "lstm_1", "trainable": true, "dtype": "float32", "return_sequences": true, "return_state": true, "go_backwards": false, "stateful": false, "unroll": false, "time_major": false, "units": 256, "activation": "tanh", "recurrent_activation": "sigmoid", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "recurrent_initializer": {"class_name": "Orthogonal", "config": {"gain": 1.0, "seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "unit_forget_bias": true, "kernel_regularizer": null, "recurrent_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "recurrent_constraint": null, "bias_constraint": null, "dropout": 0.0, "recurrent_dropout": 0.1, "implementation": 1}, "input_spec": [{"class_name": "InputSpec", "config": {"dtype": null, "shape": {"class_name": "__tuple__", "items": [null, null, 256]}, "ndim": 3, "max_ndim": null, "min_ndim": null, "axes": {}}}], "build_input_shape": [{"class_name": "TensorShape", "items": [null, null, 256]}, {"class_name": "TensorShape", "items": [null, 256]}, {"class_name": "TensorShape", "items": [null, 256]}]}
÷

!kernel
"bias
#regularization_losses
$	variables
%trainable_variables
&	keras_api
]__call__
*^&call_and_return_all_conditional_losses"Ò
_tf_keras_layer¸{"class_name": "Dense", "name": "dense", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "dense", "trainable": true, "dtype": "float32", "units": 55, "activation": "softmax", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 256}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, null, 256]}}
 "
trackable_list_wrapper
J
0
'1
(2
)3
!4
"5"
trackable_list_wrapper
J
0
'1
(2
)3
!4
"5"
trackable_list_wrapper
Ê
*layer_regularization_losses
+non_trainable_variables
	regularization_losses
,metrics

	variables
trainable_variables

-layers
.layer_metrics
S__call__
R_default_save_signature
*T&call_and_return_all_conditional_losses
&T"call_and_return_conditional_losses"
_generic_user_object
,
_serving_default"
signature_map
):'	72embedding_1/embeddings
 "
trackable_list_wrapper
'
0"
trackable_list_wrapper
'
0"
trackable_list_wrapper
­
/layer_regularization_losses
0non_trainable_variables
regularization_losses
1metrics
	variables
trainable_variables

2layers
3layer_metrics
U__call__
*V&call_and_return_all_conditional_losses
&V"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
­
4layer_regularization_losses
5non_trainable_variables
regularization_losses
6metrics
	variables
trainable_variables

7layers
8layer_metrics
W__call__
*X&call_and_return_all_conditional_losses
&X"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
­
9layer_regularization_losses
:non_trainable_variables
regularization_losses
;metrics
	variables
trainable_variables

<layers
=layer_metrics
Y__call__
*Z&call_and_return_all_conditional_losses
&Z"call_and_return_conditional_losses"
_generic_user_object
©

'kernel
(recurrent_kernel
)bias
>regularization_losses
?	variables
@trainable_variables
A	keras_api
`__call__
*a&call_and_return_all_conditional_losses"î
_tf_keras_layerÔ{"class_name": "LSTMCell", "name": "lstm_cell_1", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "lstm_cell_1", "trainable": true, "dtype": "float32", "units": 256, "activation": "tanh", "recurrent_activation": "sigmoid", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "recurrent_initializer": {"class_name": "Orthogonal", "config": {"gain": 1.0, "seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "unit_forget_bias": true, "kernel_regularizer": null, "recurrent_regularizer": null, "bias_regularizer": null, "kernel_constraint": null, "recurrent_constraint": null, "bias_constraint": null, "dropout": 0.0, "recurrent_dropout": 0.1, "implementation": 1}}
 "
trackable_list_wrapper
 "
trackable_list_wrapper
5
'0
(1
)2"
trackable_list_wrapper
5
'0
(1
)2"
trackable_list_wrapper
¹
Blayer_regularization_losses
Cnon_trainable_variables
regularization_losses
Dmetrics
	variables
trainable_variables

Estates

Flayers
Glayer_metrics
[__call__
*\&call_and_return_all_conditional_losses
&\"call_and_return_conditional_losses"
_generic_user_object
:	72dense/kernel
:72
dense/bias
 "
trackable_list_wrapper
.
!0
"1"
trackable_list_wrapper
.
!0
"1"
trackable_list_wrapper
­
Hlayer_regularization_losses
Inon_trainable_variables
#regularization_losses
Jmetrics
$	variables
%trainable_variables

Klayers
Llayer_metrics
]__call__
*^&call_and_return_all_conditional_losses
&^"call_and_return_conditional_losses"
_generic_user_object
-:+
2lstm_1/lstm_cell_1/kernel
7:5
2#lstm_1/lstm_cell_1/recurrent_kernel
&:$2lstm_1/lstm_cell_1/bias
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
X
0
1
2
3
4
5
6
7"
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
5
'0
(1
)2"
trackable_list_wrapper
5
'0
(1
)2"
trackable_list_wrapper
­
Mlayer_regularization_losses
Nnon_trainable_variables
>regularization_losses
Ometrics
?	variables
@trainable_variables

Players
Qlayer_metrics
`__call__
*a&call_and_return_all_conditional_losses
&a"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
'
0"
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
´2±
 __inference__wrapped_model_13665
²
FullArgSpec
args 
varargsjargs
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *|¢y
wt
*'
input_2ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
"
input_4ÿÿÿÿÿÿÿÿÿ
"
input_3ÿÿÿÿÿÿÿÿÿ
ê2ç
'__inference_model_2_layer_call_fn_15235
'__inference_model_2_layer_call_fn_16017
'__inference_model_2_layer_call_fn_15287
'__inference_model_2_layer_call_fn_15994À
·²³
FullArgSpec1
args)&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults
p 

 

kwonlyargs 
kwonlydefaultsª 
annotationsª *
 
Ö2Ó
B__inference_model_2_layer_call_and_return_conditional_losses_15677
B__inference_model_2_layer_call_and_return_conditional_losses_15971
B__inference_model_2_layer_call_and_return_conditional_losses_15153
B__inference_model_2_layer_call_and_return_conditional_losses_15182À
·²³
FullArgSpec1
args)&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults
p 

 

kwonlyargs 
kwonlydefaultsª 
annotationsª *
 
Õ2Ò
+__inference_embedding_1_layer_call_fn_16034¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
ð2í
F__inference_embedding_1_layer_call_and_return_conditional_losses_16027¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
Ö2Ó
,__inference_activation_1_layer_call_fn_16044¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
ñ2î
G__inference_activation_1_layer_call_and_return_conditional_losses_16039¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
2
)__inference_dropout_1_layer_call_fn_16066
)__inference_dropout_1_layer_call_fn_16071´
«²§
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaultsª 
annotationsª *
 
Æ2Ã
D__inference_dropout_1_layer_call_and_return_conditional_losses_16056
D__inference_dropout_1_layer_call_and_return_conditional_losses_16061´
«²§
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaultsª 
annotationsª *
 
û2ø
&__inference_lstm_1_layer_call_fn_16651
&__inference_lstm_1_layer_call_fn_17237
&__inference_lstm_1_layer_call_fn_17255
&__inference_lstm_1_layer_call_fn_16636Õ
Ì²È
FullArgSpecB
args:7
jself
jinputs
jmask

jtraining
jinitial_state
varargs
 
varkw
 
defaults

 
p 

 

kwonlyargs 
kwonlydefaultsª 
annotationsª *
 
ç2ä
A__inference_lstm_1_layer_call_and_return_conditional_losses_16967
A__inference_lstm_1_layer_call_and_return_conditional_losses_17219
A__inference_lstm_1_layer_call_and_return_conditional_losses_16378
A__inference_lstm_1_layer_call_and_return_conditional_losses_16621Õ
Ì²È
FullArgSpecB
args:7
jself
jinputs
jmask

jtraining
jinitial_state
varargs
 
varkw
 
defaults

 
p 

 

kwonlyargs 
kwonlydefaultsª 
annotationsª *
 
Ï2Ì
%__inference_dense_layer_call_fn_17301¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
ê2ç
@__inference_dense_layer_call_and_return_conditional_losses_17292¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
ÚB×
#__inference_signature_wrapper_15312input_2input_3input_4"
²
FullArgSpec
args 
varargs
 
varkwjkwargs
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
2
+__inference_lstm_cell_1_layer_call_fn_17521
+__inference_lstm_cell_1_layer_call_fn_17504¾
µ²±
FullArgSpec3
args+(
jself
jinputs
jstates

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaultsª 
annotationsª *
 
Ô2Ñ
F__inference_lstm_cell_1_layer_call_and_return_conditional_losses_17410
F__inference_lstm_cell_1_layer_call_and_return_conditional_losses_17487¾
µ²±
FullArgSpec3
args+(
jself
jinputs
jstates

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaultsª 
annotationsª *
 Ó
 __inference__wrapped_model_13665®')(!"¢
|¢y
wt
*'
input_2ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
"
input_4ÿÿÿÿÿÿÿÿÿ
"
input_3ÿÿÿÿÿÿÿÿÿ
ª "ª
5
dense,)
denseÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ7
+
lstm_1!
lstm_1ÿÿÿÿÿÿÿÿÿ
/
lstm_1_1# 
lstm_1_1ÿÿÿÿÿÿÿÿÿ¿
G__inference_activation_1_layer_call_and_return_conditional_losses_16039t=¢:
3¢0
.+
inputsÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
ª "3¢0
)&
0ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 
,__inference_activation_1_layer_call_fn_16044g=¢:
3¢0
.+
inputsÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
ª "&#ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ»
@__inference_dense_layer_call_and_return_conditional_losses_17292w!"=¢:
3¢0
.+
inputsÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
ª "2¢/
(%
0ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ7
 
%__inference_dense_layer_call_fn_17301j!"=¢:
3¢0
.+
inputsÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
ª "%"ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ7À
D__inference_dropout_1_layer_call_and_return_conditional_losses_16056xA¢>
7¢4
.+
inputsÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
p
ª "3¢0
)&
0ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 À
D__inference_dropout_1_layer_call_and_return_conditional_losses_16061xA¢>
7¢4
.+
inputsÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
p 
ª "3¢0
)&
0ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 
)__inference_dropout_1_layer_call_fn_16066kA¢>
7¢4
.+
inputsÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
p
ª "&#ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
)__inference_dropout_1_layer_call_fn_16071kA¢>
7¢4
.+
inputsÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
p 
ª "&#ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ¼
F__inference_embedding_1_layer_call_and_return_conditional_losses_16027r8¢5
.¢+
)&
inputsÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
ª "3¢0
)&
0ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 
+__inference_embedding_1_layer_call_fn_16034e8¢5
.¢+
)&
inputsÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
ª "&#ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
A__inference_lstm_1_layer_call_and_return_conditional_losses_16378Ó')(P¢M
F¢C
52
0-
inputs/0ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ

 
p

 
ª "z¢w
pm
+(
0/0ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ

0/1ÿÿÿÿÿÿÿÿÿ

0/2ÿÿÿÿÿÿÿÿÿ
 
A__inference_lstm_1_layer_call_and_return_conditional_losses_16621Ó')(P¢M
F¢C
52
0-
inputs/0ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ

 
p 

 
ª "z¢w
pm
+(
0/0ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ

0/1ÿÿÿÿÿÿÿÿÿ

0/2ÿÿÿÿÿÿÿÿÿ
 
A__inference_lstm_1_layer_call_and_return_conditional_losses_16967Î')(Ê¢Æ
¾¢º
.+
inputsÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
'$
maskÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ

p
[X
*'
initial_state/0ÿÿÿÿÿÿÿÿÿ
*'
initial_state/1ÿÿÿÿÿÿÿÿÿ
ª "z¢w
pm
+(
0/0ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ

0/1ÿÿÿÿÿÿÿÿÿ

0/2ÿÿÿÿÿÿÿÿÿ
 
A__inference_lstm_1_layer_call_and_return_conditional_losses_17219Î')(Ê¢Æ
¾¢º
.+
inputsÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
'$
maskÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ

p 
[X
*'
initial_state/0ÿÿÿÿÿÿÿÿÿ
*'
initial_state/1ÿÿÿÿÿÿÿÿÿ
ª "z¢w
pm
+(
0/0ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ

0/1ÿÿÿÿÿÿÿÿÿ

0/2ÿÿÿÿÿÿÿÿÿ
 î
&__inference_lstm_1_layer_call_fn_16636Ã')(P¢M
F¢C
52
0-
inputs/0ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ

 
p

 
ª "jg
)&
0ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ

1ÿÿÿÿÿÿÿÿÿ

2ÿÿÿÿÿÿÿÿÿî
&__inference_lstm_1_layer_call_fn_16651Ã')(P¢M
F¢C
52
0-
inputs/0ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ

 
p 

 
ª "jg
)&
0ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ

1ÿÿÿÿÿÿÿÿÿ

2ÿÿÿÿÿÿÿÿÿé
&__inference_lstm_1_layer_call_fn_17237¾')(Ê¢Æ
¾¢º
.+
inputsÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
'$
maskÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ

p
[X
*'
initial_state/0ÿÿÿÿÿÿÿÿÿ
*'
initial_state/1ÿÿÿÿÿÿÿÿÿ
ª "jg
)&
0ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ

1ÿÿÿÿÿÿÿÿÿ

2ÿÿÿÿÿÿÿÿÿé
&__inference_lstm_1_layer_call_fn_17255¾')(Ê¢Æ
¾¢º
.+
inputsÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
'$
maskÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ

p 
[X
*'
initial_state/0ÿÿÿÿÿÿÿÿÿ
*'
initial_state/1ÿÿÿÿÿÿÿÿÿ
ª "jg
)&
0ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ

1ÿÿÿÿÿÿÿÿÿ

2ÿÿÿÿÿÿÿÿÿÏ
F__inference_lstm_cell_1_layer_call_and_return_conditional_losses_17410')(¢
y¢v
!
inputsÿÿÿÿÿÿÿÿÿ
M¢J
# 
states/0ÿÿÿÿÿÿÿÿÿ
# 
states/1ÿÿÿÿÿÿÿÿÿ
p
ª "v¢s
l¢i

0/0ÿÿÿÿÿÿÿÿÿ
GD
 
0/1/0ÿÿÿÿÿÿÿÿÿ
 
0/1/1ÿÿÿÿÿÿÿÿÿ
 Ï
F__inference_lstm_cell_1_layer_call_and_return_conditional_losses_17487')(¢
y¢v
!
inputsÿÿÿÿÿÿÿÿÿ
M¢J
# 
states/0ÿÿÿÿÿÿÿÿÿ
# 
states/1ÿÿÿÿÿÿÿÿÿ
p 
ª "v¢s
l¢i

0/0ÿÿÿÿÿÿÿÿÿ
GD
 
0/1/0ÿÿÿÿÿÿÿÿÿ
 
0/1/1ÿÿÿÿÿÿÿÿÿ
 ¤
+__inference_lstm_cell_1_layer_call_fn_17504ô')(¢
y¢v
!
inputsÿÿÿÿÿÿÿÿÿ
M¢J
# 
states/0ÿÿÿÿÿÿÿÿÿ
# 
states/1ÿÿÿÿÿÿÿÿÿ
p
ª "f¢c

0ÿÿÿÿÿÿÿÿÿ
C@

1/0ÿÿÿÿÿÿÿÿÿ

1/1ÿÿÿÿÿÿÿÿÿ¤
+__inference_lstm_cell_1_layer_call_fn_17521ô')(¢
y¢v
!
inputsÿÿÿÿÿÿÿÿÿ
M¢J
# 
states/0ÿÿÿÿÿÿÿÿÿ
# 
states/1ÿÿÿÿÿÿÿÿÿ
p 
ª "f¢c

0ÿÿÿÿÿÿÿÿÿ
C@

1/0ÿÿÿÿÿÿÿÿÿ

1/1ÿÿÿÿÿÿÿÿÿÞ
B__inference_model_2_layer_call_and_return_conditional_losses_15153')(!"¢
¢
wt
*'
input_2ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
"
input_4ÿÿÿÿÿÿÿÿÿ
"
input_3ÿÿÿÿÿÿÿÿÿ
p

 
ª "y¢v
ol
*'
0/0ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ7

0/1ÿÿÿÿÿÿÿÿÿ

0/2ÿÿÿÿÿÿÿÿÿ
 Þ
B__inference_model_2_layer_call_and_return_conditional_losses_15182')(!"¢
¢
wt
*'
input_2ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
"
input_4ÿÿÿÿÿÿÿÿÿ
"
input_3ÿÿÿÿÿÿÿÿÿ
p 

 
ª "y¢v
ol
*'
0/0ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ7

0/1ÿÿÿÿÿÿÿÿÿ

0/2ÿÿÿÿÿÿÿÿÿ
 á
B__inference_model_2_layer_call_and_return_conditional_losses_15677')(!"¢
¢
zw
+(
inputs/0ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
# 
inputs/1ÿÿÿÿÿÿÿÿÿ
# 
inputs/2ÿÿÿÿÿÿÿÿÿ
p

 
ª "y¢v
ol
*'
0/0ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ7

0/1ÿÿÿÿÿÿÿÿÿ

0/2ÿÿÿÿÿÿÿÿÿ
 á
B__inference_model_2_layer_call_and_return_conditional_losses_15971')(!"¢
¢
zw
+(
inputs/0ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
# 
inputs/1ÿÿÿÿÿÿÿÿÿ
# 
inputs/2ÿÿÿÿÿÿÿÿÿ
p 

 
ª "y¢v
ol
*'
0/0ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ7

0/1ÿÿÿÿÿÿÿÿÿ

0/2ÿÿÿÿÿÿÿÿÿ
 ³
'__inference_model_2_layer_call_fn_15235')(!"¢
¢
wt
*'
input_2ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
"
input_4ÿÿÿÿÿÿÿÿÿ
"
input_3ÿÿÿÿÿÿÿÿÿ
p

 
ª "if
(%
0ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ7

1ÿÿÿÿÿÿÿÿÿ

2ÿÿÿÿÿÿÿÿÿ³
'__inference_model_2_layer_call_fn_15287')(!"¢
¢
wt
*'
input_2ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
"
input_4ÿÿÿÿÿÿÿÿÿ
"
input_3ÿÿÿÿÿÿÿÿÿ
p 

 
ª "if
(%
0ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ7

1ÿÿÿÿÿÿÿÿÿ

2ÿÿÿÿÿÿÿÿÿ¶
'__inference_model_2_layer_call_fn_15994')(!"¢
¢
zw
+(
inputs/0ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
# 
inputs/1ÿÿÿÿÿÿÿÿÿ
# 
inputs/2ÿÿÿÿÿÿÿÿÿ
p

 
ª "if
(%
0ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ7

1ÿÿÿÿÿÿÿÿÿ

2ÿÿÿÿÿÿÿÿÿ¶
'__inference_model_2_layer_call_fn_16017')(!"¢
¢
zw
+(
inputs/0ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
# 
inputs/1ÿÿÿÿÿÿÿÿÿ
# 
inputs/2ÿÿÿÿÿÿÿÿÿ
p 

 
ª "if
(%
0ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ7

1ÿÿÿÿÿÿÿÿÿ

2ÿÿÿÿÿÿÿÿÿô
#__inference_signature_wrapper_15312Ì')(!"¥¢¡
¢ 
ª
5
input_2*'
input_2ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
-
input_3"
input_3ÿÿÿÿÿÿÿÿÿ
-
input_4"
input_4ÿÿÿÿÿÿÿÿÿ"ª
5
dense,)
denseÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ7
+
lstm_1!
lstm_1ÿÿÿÿÿÿÿÿÿ
/
lstm_1_1# 
lstm_1_1ÿÿÿÿÿÿÿÿÿ