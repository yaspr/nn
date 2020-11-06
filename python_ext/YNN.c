//
#include <Python.h>

//
#include "YNN_core.h"

//
static PyObject *YNN_error = NULL;

//
static PyObject *YNN_version(PyObject *self)
{
  return Py_BuildValue("s", "YNN version 0.1");
}

//
static PyObject *YNN_sigmoid(PyObject *self, PyObject *args)
{
  float x = 0.0;
  
  if (!PyArg_ParseTuple(args, "f", &x))
    return NULL;

  return Py_BuildValue("f", sigmoid(x));
}

//
static PyObject *YNN_d_sigmoid(PyObject *self, PyObject *args)
{
  float x = 0.0;
  
  if (!PyArg_ParseTuple(args, "f", &x))
    return NULL;

  return Py_BuildValue("f", d_sigmoid(x));
}

//Python wrapper for the reduc_f32 C function
static PyObject *YNN_reduc_f32(PyObject *self, PyObject *args)
{
  Py_buffer view;
  PyObject *obj = NULL;

  //Get the parameter (expected: 1-dimensional array)
  if (!PyArg_ParseTuple(args, "O", &obj))
    return NULL;

  //Get the array memory view
  if (PyObject_GetBuffer(obj, &view, PyBUF_ANY_CONTIGUOUS | PyBUF_FORMAT) == -1)
    return NULL;

  //
  if (view.ndim != 1)
    {
      PyErr_SetString(PyExc_TypeError, "Expecting a 1-dimensional array.");
      PyBuffer_Release(&view);
      return NULL;
    }

  //
  if (strcmp(view.format, "f"))
    {
      PyErr_SetString(PyExc_TypeError, "Expecting a 1-dimensional array of floats.");
      PyBuffer_Release(&view);
      return NULL;     
    }
  
  //Return the C function's return value as a Python float object 
  return Py_BuildValue("f", reduc_f32(view.buf, view.shape[0]));
}

//Python wrapper for the reduc_f32 C function
static PyObject *YNN_reduc_f32_AVX(PyObject *self, PyObject *args)
{
  Py_buffer view;
  PyObject *obj = NULL;

  //Get the parameter (expected: 1-dimensional array)
  if (!PyArg_ParseTuple(args, "O", &obj))
    return NULL;

  //Get the array memory view
  if (PyObject_GetBuffer(obj, &view, PyBUF_ANY_CONTIGUOUS | PyBUF_FORMAT) == -1)
    return NULL;

  //
  if (view.ndim != 1)
    {
      PyErr_SetString(PyExc_TypeError, "Expecting a 1-dimensional array.");
      PyBuffer_Release(&view);
      return NULL;
    }

  //
  if (strcmp(view.format, "f"))
    {
      PyErr_SetString(PyExc_TypeError, "Expecting a 1-dimensional array of floats.");
      PyBuffer_Release(&view);
      return NULL;     
    }

  //Return the C function's return value as a Python float object 
  return Py_BuildValue("f", reduc_f32_AVX(view.buf, view.shape[0]));
}

//Python wrapper for the dotprod_f32 C function
static PyObject *YNN_dotprod_f32(PyObject *self, PyObject *args)
{
  //Views and objects for the parameter arrays
  Py_buffer view1;
  Py_buffer view2;
  PyObject *obj1 = NULL;
  PyObject *obj2 = NULL;

  //Get the parameters (2 1-dimensional arrays)
  if (!PyArg_ParseTuple(args, "OO", &obj1, &obj2))
    return NULL;

  //Get the first array memory view
  if (PyObject_GetBuffer(obj1, &view1, PyBUF_ANY_CONTIGUOUS | PyBUF_FORMAT) == -1)
    return NULL;

  //Get the second array memory view
  if (PyObject_GetBuffer(obj2, &view2, PyBUF_ANY_CONTIGUOUS | PyBUF_FORMAT) == -1)
    return NULL;

  //
  if (view1.ndim != 1 || view2.ndim != 1)
    {
      PyErr_SetString(PyExc_TypeError, "Expecting a 1-dimensional array.");
      PyBuffer_Release(&view1);
      PyBuffer_Release(&view2);
      return NULL;
    }

  //
  if (strcmp(view1.format, "f") || strcmp(view2.format, "f"))
    {
      PyErr_SetString(PyExc_TypeError, "Expecting a 1-dimensional array of floats.");
      PyBuffer_Release(&view1);
      PyBuffer_Release(&view2);
           return NULL;     
    }

  //
  if (view1.shape[0] != view2.shape[0])
    {
      PyErr_SetString(PyExc_TypeError, "Expecting two 1-dimensional arrays of the same size.");
      PyBuffer_Release(&view1);
      PyBuffer_Release(&view2);
      return NULL;     
    }

  //Return the C function's return value as a Python float object
  return Py_BuildValue("f", dotprod_f32(view1.buf, view2.buf, view1.shape[0]));
}

//Python wrapper for the dotprod_f32 C function
static PyObject *YNN_dotprod_f32_AVX(PyObject *self, PyObject *args)
{
  //Views and objects for the parameter arrays
  Py_buffer view1;
  Py_buffer view2;
  PyObject *obj1 = NULL;
  PyObject *obj2 = NULL;

  //Get the parameters (2 1-dimensional arrays)
  if (!PyArg_ParseTuple(args, "OO", &obj1, &obj2))
    return NULL;

  //Get the first array memory view
  if (PyObject_GetBuffer(obj1, &view1, PyBUF_ANY_CONTIGUOUS | PyBUF_FORMAT) == -1)
    return NULL;

  //Get the second array memory view
  if (PyObject_GetBuffer(obj2, &view2, PyBUF_ANY_CONTIGUOUS | PyBUF_FORMAT) == -1)
    return NULL;

  //
  if (view1.ndim != 1 || view2.ndim != 1)
    {
      PyErr_SetString(PyExc_TypeError, "Expecting a 1-dimensional array.");
      PyBuffer_Release(&view1);
      PyBuffer_Release(&view2);
      return NULL;
    }

  //
  if (strcmp(view1.format, "f") || strcmp(view2.format, "f"))
    {
      PyErr_SetString(PyExc_TypeError, "Expecting a 1-dimensional array of floats.");
      PyBuffer_Release(&view1);
      PyBuffer_Release(&view2);
           return NULL;     
    }

  //
  if (view1.shape[0] != view2.shape[0])
    {
      PyErr_SetString(PyExc_TypeError, "Expecting two 1-dimensional arrays of the same size.");
      PyBuffer_Release(&view1);
      PyBuffer_Release(&view2);
      return NULL;     
    }

  //Return the C function's return value as a Python float object
  return Py_BuildValue("f", dotprod_f32_AVX(view1.buf, view2.buf, view1.shape[0]));
}

//Register the methods to be made available Python side
static PyMethodDef YNN_methods[] = {

  //
  { "sigmoid"     , YNN_sigmoid    , METH_VARARGS, "Returns the Sigmoid value of the given float parameter."},
  { "d_sigmoid"   , YNN_d_sigmoid  , METH_VARARGS, "Returns the Sigmoid derivative of the given float parameter."},
  { "reduc_f32"   , YNN_reduc_f32  , METH_VARARGS, "Returns the sum of all the float elements of the given array."},
  { "dotprod_f32" , YNN_dotprod_f32, METH_VARARGS, "Returns the dotproduct of all the float elements of the given arrays."},
  { "reduc_f32_optimized"   , YNN_reduc_f32_AVX  , METH_VARARGS, "Returns the sum of all the float elements of the given array (optimized for x86 with AVX)."},
  { "dotprod_f32_optimized" , YNN_dotprod_f32_AVX, METH_VARARGS, "Returns the dotproduct of all the float elements of the given arrays (optimized for x86 with AVX)."},
  
  //
  { "version", (PyCFunction)YNN_version, METH_VARARGS, "Returns the version of the YNN library." },
  
  { NULL, NULL, 0, NULL}
};

//
static PyModuleDef YNN_module = {
  
  PyModuleDef_HEAD_INIT,
  "YNN",
  "YNN neural networks library",
  -1,
  YNN_methods
};

//
PyMODINIT_FUNC PyInit_YNN()
{
  PyObject *obj = PyModule_Create(&YNN_module);

  if (!obj)
    return NULL;

  YNN_error = PyErr_NewException("YNN.error", NULL, NULL);
  Py_XINCREF(YNN_error);
  
  if (PyModule_AddObject(obj, "error", YNN_error) < 0)
    {
      Py_XDECREF(YNN_error);
      Py_CLEAR(YNN_error);
      Py_DECREF(obj);
      return NULL;
    }

  return obj;
}
