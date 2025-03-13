Time Windows
============

Definition
-----------

Time Windows in VERUS define periods when the vulnerability influence (VI) of PoTIs changes. They allow for modeling the temporal dynamics of urban environments where population distribution and activity patterns shift throughout the day.

In current development stage, time windows are defined as a dictionary with the following structure:

.. code-block:: python

    {
        "category": pd.DataFrame({
            "ts": [start_time_unix_epoch],
            "te": [end_time_unix_epoch],
            "vi": [vi_value]
        })
    }

Where: 

* `category` is the PoTI category to which the time window applies
* `ts` is the start time of the window in Unix epoch format
* `te` is the end time of the window in Unix epoch format
* `vi` is the vulnerability influence value during this time period

We decided to use epoch format for time representation to avoid timezone issues and streamline calculations. This also allows to define specific time windows with higher precision and computationally analyze them more efficiently.

In this current form, its intended to be used as a reference for future development of this feature.

Purpose
-------

Time windows enable:

* Modeling scenarios at different times of day (rush hour, school hours, etc.)
* Accounting for locations that are only active during specific hours
* Creating time-aware vulnerability assessments
* Comparing vulnerability patterns across different temporal contexts

Structure
---------

A time window contains:

1. **Start time (ts)**: When the specific vulnerability influence begins
2. **End time (te)**: When the influence period ends
3. **Category**: Which type of PoTI this window applies to
4. **VI value**: The vulnerability influence value during this time period

Example
-------

A school might have the following time windows:

* 08:00-15:00: VI = 0.9 (school hours, high vulnerability)
* 15:00-17:00: VI = 0.5 (after-school activities)
* 17:00-08:00: VI = 0.1 (closed, minimal vulnerability)

This reflects how the presence of students and staff makes a school a higher-vulnerability location during operating hours.

Generation
----------

Time windows can be generated using the `TimeWindowGenerator`:

.. code-block:: python

    tw_gen = TimeWindowGenerator()
    
    # Generate from predefined schedules
    time_windows = tw_gen.generate_from_schedule()
    
    # Or create custom time windows
    custom_windows = {
        "school": pd.DataFrame({
            "ts": [tw_gen.to_unix_epoch("08:00")],
            "te": [tw_gen.to_unix_epoch("15:00")],
            "vi": [0.9]
        })
    }

Integration with VERUS
----------------------

Time windows are applied during the vulnerability assessment process:

1. For a given evaluation time, VERUS identifies which time window is active for each PoTI category
2. The VI value from the applicable time window is applied to each PoTI
3. These updated VI values are then used in vulnerability calculations for a given evaluation time

This dynamic approach allows for more realistic modeling of urban environments where vulnerability isn't static but changes with human activity patterns.