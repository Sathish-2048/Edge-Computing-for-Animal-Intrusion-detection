from onnx_tf.backend import prepare
import onnx

onnx_model = onnx.load("best.onnx")
tf_rep = prepare(onnx_model)
tf_rep.export_graph("saved_model")  # Creates folder 'saved_model'
