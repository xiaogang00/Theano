#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sun Mar  5 16:12:53 2017

@author: root
"""
import theano
import theano.tensor as T

x = T.dvector('x')
y = x ** 2
cost = y.sum()
gy = T.grad(cost, x)
H, updates = theano.scan(lambda i, gy,x : T.grad(gy[i], x), sequences=T.arange(gy.shape[0]), non_sequences=[gy, x])
f = theano.function([x], H, updates=updates)
print f([4, 4])
#array([[ 2.,  0.],
#       [ 0.,  2.]])
"""
Help on function scan in theano:

theano.scan = scan(fn, sequences=None, outputs_info=None, non_sequences=None, n_steps=None, truncate_gradient=-1, go_backwards=False, mode=None, name=None, profile=False, allow_gc=None, strict=False)
    This function constructs and applies a Scan op to the provided
    arguments.
    
    Parameters
    ----------
    fn
        ``fn`` is a function that describes the operations involved in one
        step of ``scan``. ``fn`` should construct variables describing the
        output of one iteration step. It should expect as input theano
        variables representing all the slices of the input sequences
        and previous values of the outputs, as well as all other arguments
        given to scan as ``non_sequences``. The order in which scan passes
        these variables to ``fn``  is the following :
    
        * all time slices of the first sequence
        * all time slices of the second sequence
        * ...
        * all time slices of the last sequence
        * all past slices of the first output
        * all past slices of the second otuput
        * ...
        * all past slices of the last output
        * all other arguments (the list given as `non_sequences` to
            scan)
    
        The order of the sequences is the same as the one in the list
        `sequences` given to scan. The order of the outputs is the same
        as the order of ``outputs_info``. For any sequence or output the
        order of the time slices is the same as the one in which they have
        been given as taps. For example if one writes the following :
    
        .. code-block:: python
    
            scan(fn, sequences = [ dict(input= Sequence1, taps = [-3,2,-1])
                                 , Sequence2
                                 , dict(input =  Sequence3, taps = 3) ]
                   , outputs_info = [ dict(initial =  Output1, taps = [-3,-5])
                                    , dict(initial = Output2, taps = None)
                                    , Output3 ]
                   , non_sequences = [ Argument1, Argument2])
    
        ``fn`` should expect the following arguments in this given order:
    
        #. ``Sequence1[t-3]``
        #. ``Sequence1[t+2]``
        #. ``Sequence1[t-1]``
        #. ``Sequence2[t]``
        #. ``Sequence3[t+3]``
        #. ``Output1[t-3]``
        #. ``Output1[t-5]``
        #. ``Output3[t-1]``
        #. ``Argument1``
        #. ``Argument2``
    
        The list of ``non_sequences`` can also contain shared variables
        used in the function, though ``scan`` is able to figure those
        out on its own so they can be skipped. For the clarity of the
        code we recommend though to provide them to scan. To some extend
        ``scan`` can also figure out other ``non sequences`` (not shared)
        even if not passed to scan (but used by `fn`). A simple example of
        this would be :
    
        .. code-block:: python
    
            import theano.tensor as TT
            W   = TT.matrix()
            W_2 = W**2
            def f(x):
                return TT.dot(x,W_2)
    
        The function is expected to return two things. One is a list of
        outputs ordered in the same order as ``outputs_info``, with the
        difference that there should be only one output variable per
        output initial state (even if no tap value is used). Secondly
        `fn` should return an update dictionary (that tells how to
        update any shared variable after each iteration step). The
        dictionary can optionally be given as a list of tuples. There is
        no constraint on the order of these two list, ``fn`` can return
        either ``(outputs_list, update_dictionary)`` or
        ``(update_dictionary, outputs_list)`` or just one of the two (in
        case the other is empty).
    
        To use ``scan`` as a while loop, the user needs to change the
        function ``fn`` such that also a stopping condition is returned.
        To do so, he/she needs to wrap the condition in an ``until`` class.
        The condition should be returned as a third element, for example:
    
        .. code-block:: python
    
            ...
            return [y1_t, y2_t], {x:x+1}, theano.scan_module.until(x < 50)
    
        Note that a number of steps (considered in here as the maximum
        number of steps ) is still required even though a condition is
        passed (and it is used to allocate memory if needed). = {}):
    
    sequences
        ``sequences`` is the list of Theano variables or dictionaries
        describing the sequences ``scan`` has to iterate over. If a
        sequence is given as wrapped in a dictionary, then a set of optional
        information can be provided about the sequence. The dictionary
        should have the following keys:
    
        * ``input`` (*mandatory*) -- Theano variable representing the
          sequence.
    
        * ``taps`` -- Temporal taps of the sequence required by ``fn``.
          They are provided as a list of integers, where a value ``k``
          impiles that at iteration step ``t`` scan will pass to ``fn``
          the slice ``t+k``. Default value is ``[0]``
    
        Any Theano variable in the list ``sequences`` is automatically
        wrapped into a dictionary where ``taps`` is set to ``[0]``
    
    outputs_info
        ``outputs_info`` is the list of Theano variables or dictionaries
        describing the initial state of the outputs computed
        recurrently. When this initial states are given as dictionary
        optional information can be provided about the output corresponding
        to these initial states. The dictionary should have the following
        keys:
    
        * ``initial`` -- Theano variable that represents the initial
          state of a given output. In case the output is not computed
          recursively (think of a map) and does not require an initial
          state this field can be skipped. Given that (only) the previous
          time step of the output is used by ``fn``, the initial state
          **should have the same shape** as the output and **should not
          involve a downcast** of the data type of the output. If multiple
          time taps are used, the initial state should have one extra
          dimension that should cover all the possible taps. For example
          if we use ``-5``, ``-2`` and ``-1`` as past taps, at step 0,
          ``fn`` will require (by an abuse of notation) ``output[-5]``,
          ``output[-2]`` and ``output[-1]``. This will be given by
          the initial state, which in this case should have the shape
          (5,)+output.shape. If this variable containing the initial
          state is called ``init_y`` then ``init_y[0]`` *corresponds to*
          ``output[-5]``. ``init_y[1]`` *correponds to* ``output[-4]``,
          ``init_y[2]`` corresponds to ``output[-3]``, ``init_y[3]``
          coresponds to ``output[-2]``, ``init_y[4]`` corresponds to
          ``output[-1]``. While this order might seem strange, it comes
          natural from splitting an array at a given point. Assume that
          we have a array ``x``, and we choose ``k`` to be time step
          ``0``. Then our initial state would be ``x[:k]``, while the
          output will be ``x[k:]``. Looking at this split, elements in
          ``x[:k]`` are ordered exactly like those in ``init_y``.
        * ``taps`` -- Temporal taps of the output that will be pass to
          ``fn``. They are provided as a list of *negative* integers,
          where a value ``k`` implies that at iteration step ``t`` scan
          will pass to ``fn`` the slice ``t+k``.
    
        ``scan`` will follow this logic if partial information is given:
    
        * If an output is not wrapped in a dictionary, ``scan`` will wrap
          it in one assuming that you use only the last step of the output
          (i.e. it makes your tap value list equal to [-1]).
        * If you wrap an output in a dictionary and you do not provide any
          taps but you provide an initial state it will assume that you are
          using only a tap value of -1.
        * If you wrap an output in a dictionary but you do not provide any
          initial state, it assumes that you are not using any form of
          taps.
        * If you provide a ``None`` instead of a variable or a empty
          dictionary ``scan`` assumes that you will not use any taps for
          this output (like for example in case of a map)
    
        If ``outputs_info`` is an empty list or None, ``scan`` assumes
        that no tap is used for any of the outputs. If information is
        provided just for a subset of the outputs an exception is
        raised (because there is no convention on how scan should map
        the provided information to the outputs of ``fn``)
    
    non_sequences
        ``non_sequences`` is the list of arguments that are passed to
        ``fn`` at each steps. One can opt to exclude variable
        used in ``fn`` from this list as long as they are part of the
        computational graph, though for clarity we encourage not to do so.
    
    n_steps
        ``n_steps`` is the number of steps to iterate given as an int
        or Theano scalar. If any of the input sequences do not have
        enough elements, scan will raise an error. If the *value is 0* the
        outputs will have *0 rows*. If the value is negative, ``scan``
        will run backwards in time. If the ``go_backwards`` flag is already
        set and also ``n_steps`` is negative, ``scan`` will run forward
        in time. If n_steps is not provided, ``scan`` will figure
        out the amount of steps it should run given its input sequences.
    
    truncate_gradient
        ``truncate_gradient`` is the number of steps to use in truncated
        BPTT.  If you compute gradients through a scan op, they are
        computed using backpropagation through time. By providing a
        different value then -1, you choose to use truncated BPTT instead
        of classical BPTT, where you go for only ``truncate_gradient``
        number of steps back in time.
    
    go_backwards
        ``go_backwards`` is a flag indicating if ``scan`` should go
        backwards through the sequences. If you think of each sequence
        as indexed by time, making this flag True would mean that
        ``scan`` goes back in time, namely that for any sequence it
        starts from the end and goes towards 0.
    
    name
        When profiling ``scan``, it is crucial to provide a name for any
        instance of ``scan``. The profiler will produce an overall
        profile of your code as well as profiles for the computation of
        one step of each instance of ``scan``. The ``name`` of the instance
        appears in those profiles and can greatly help to disambiguate
        information.
    
    mode
        It is recommended to leave this argument to None, especially
        when profiling ``scan`` (otherwise the results are not going to
        be accurate). If you prefer the computations of one step of
        ``scan`` to be done differently then the entire function, you
        can use this parameter to describe how the computations in this
        loop are done (see ``theano.function`` for details about
        possible values and their meaning).
    
    profile
        Flag or string. If true, or different from the empty string, a
        profile object will be created and attached to the inner graph of
        scan. In case ``profile`` is True, the profile object will have the
        name of the scan instance, otherwise it will have the passed string.
        Profile object collect (and print) information only when running the
        inner graph with the new cvm linker ( with default modes,
        other linkers this argument is useless)
    
    allow_gc
        Set the value of allow gc for the internal graph of scan.  If
        set to None, this will use the value of config.scan.allow_gc.
    
    strict
        If true, all the shared variables used in ``fn`` must be provided as a
        part of ``non_sequences`` or ``sequences``.
    
    Returns
    -------
    tuple
        Tuple of the form (outputs, updates); ``outputs`` is either a
        Theano variable or a list of Theano variables representing the
        outputs of ``scan`` (in the same order as in ``outputs_info``).
        ``updates`` is a subclass of dictionary specifying the update rules for
        all shared variables used in scan.
        This dictionary should be passed to ``theano.function`` when you compile
        your function. The change compared to a normal dictionary is that we
        validate that keys are SharedVariable and addition of those dictionary
        are validated to be consistent.

"""