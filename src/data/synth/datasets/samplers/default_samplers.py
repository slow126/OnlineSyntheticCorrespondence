def default_julia_sampler():
    return dict(distribution='uniform', loc=2.5, scale=3)


def default_angle_sampler():
    return dict(
        x_components=dict(distribution='uniform', loc=-0.25, scale=0.3),
        y_components=dict(distribution='uniform', loc=-0.1, scale=0.3),
        bounds=(0.5, 0.25),
    )


def default_scale_sampler():
    return dict(
        abs_components=dict(distribution='uniform', loc=0.45, scale=0.25),
        rel_components=dict(distribution='uniform', loc=-0.1, scale=0.25),
    )

def default_mandelbulb_sampler():
    return dict(
        power_range={'min': 1.0, 'offset': 10.0},
        debug=True,
    )