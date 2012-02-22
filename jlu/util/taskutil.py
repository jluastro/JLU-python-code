"""
Functions used in converting other functions into tasks.

Use: import taskutil
"""

import inspect, string, sys

def get_global_namespace(ns_label='ipython console'):
    """
    Returns a dictionary of the globals in the lowest stacklevel containing
    ns_label in its label.
    """
    a=inspect.stack()
    stacklevel=0
    for k in range(len(a)):
        if (string.find(a[k][1], 'ipython console') > 0):
            stacklevel=k
#            break             # Would this be an improvement?
    return sys._getframe(stacklevel).f_globals

def update_myf(myf):
    """
    Fills unfilled parameters with defaults, and handles globals or user
    override of arguments.
    """
    ###fill unfilled parameters with defaults
    myf['update_params'](func=myf['taskname'], printtext=False)

    #Handle globals or user over-ride of arguments
    function_signature_defaults=dict(zip(radplot.func_code.co_varnames,
                                         radplot.func_defaults))
    for item in function_signature_defaults.iteritems():
        key,val = item
        keyVal = eval(key)
        if (keyVal == None):
            pass              # user hasn't set it - use global/default
        else:
            myf[key] = keyVal # user has set it - use over-ride

def check_param_types(taskname, arg_desc):
    """
    Checks the values of taskname's arguments against their allowed types.
    Suitable for tasks without any menu parameters.
    Returns None if everything seems OK, and the exception otherwise (after
    printing an error message).
    """
    e = None
    arg_names = arg_desc.keys()
    arg_values=[]
    arg_types =[]
    for arg in arg_names:
        arg_values.append(arg_desc[arg][0])
        arg_types.append(arg_desc[arg][1])
    try:
        parameter_checktype(arg_names, arg_values, arg_types)
    except TypeError, e:
        print taskname, "-- TypeError: ", e
    except ValueError, e:
        print taskname, " -- OptionError: ", e
    return e

def startlog(casalog, taskname):
    """
    Starts an entry in casalogger for taskname.
    """
    casalog.origin(taskname)
    casalog.post('')
    casalog.post('###############################################')
    casalog.post('###  Begin Task: %s  ###' % taskname)
    return

def endlog(casalog, taskname):
    """
    Finishes an entry in casalogger for taskname.
    """
    casalog.post('###  End of task: %s  ###' % taskname)
    casalog.post('###############################################')
    casalog.post('')
    return
