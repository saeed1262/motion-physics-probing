from shapely.geometry import MultiPoint, Point

FPS = 20.0
FOOT_JOINTS = [7, 10, 8, 11]                       # L ankle, L toe, R ankle, R toe (HumanML3D order)
FOOT_HEIGHT_MAX = np.array([0.12, 0.06, 0.12, 0.06])
FOOT_DISP2_MAX = 0.002                             # HumanML3D contact rule (squared m per frame)
FOOT_RADIUS = 0.04                                 # padding around contact points (m)
SEGMENTS = [(12, 15, 0.081), (0, 3, 0.142), (3, 9, 0.355), (16, 18, 0.028), (17, 19, 0.028),
            (18, 20, 0.016), (19, 21, 0.016), (20, 20, 0.006), (21, 21, 0.006), (1, 4, 0.100),
            (2, 5, 0.100), (4, 7, 0.0465), (5, 8, 0.0465), (7, 10, 0.0145), (8, 11, 0.0145)]


def center_of_mass(j):
    com = np.zeros((j.shape[0], 3))
    for a, b, w in SEGMENTS:
        com += w * 0.5 * (j[:, a] + j[:, b])
    return com


def foot_contacts(j):
    feet = j[:, FOOT_JOINTS]
    d2 = np.zeros(feet.shape[:2])
    d2[1:] = ((feet[1:] - feet[:-1]) ** 2).sum(-1)
    d2[0] = d2[1]
    return (d2 < FOOT_DISP2_MAX) & (feet[..., 1] < FOOT_HEIGHT_MAX[None])


def balance_margin(com_xz, pts_xz):
    if len(pts_xz) == 0:
        return np.nan
    poly = MultiPoint([tuple(p) for p in pts_xz]).convex_hull.buffer(FOOT_RADIUS)
    p = Point(*com_xz)
    return poly.exterior.distance(p) if poly.contains(p) else -poly.distance(p)


MIN_FLIGHT = 4      # airborne frames needed to fit a ballistic arc (0.2 s at 20 fps)
G_EARTH = 9.81
SUPPORT_JOINTS = [0, 4, 5, 20, 21]  # pelvis, knees, wrists: body parts that can carry weight instead of the feet
SUPPORT_HEIGHT_MIN = 0.15           # any of them this close to the floor means supported, not airborne
FLOAT_G_MAX = 2.0                   # an unsupported stretch falling slower than this (m/s^2) is floating, not a jump


def flight_gravity(com_y, airborne):
    # Fit y(t) = c0 + c1 t + c2 t^2 to the center of mass over each airborne stretch.
    # Real ballistic flight gives g = -2 c2 = 9.81 m/s^2; a body that hovers gives g near 0.
    g = np.full(len(com_y), np.nan)
    t, n = 0, len(com_y)
    while t < n:
        if not airborne[t]:
            t += 1
            continue
        s = t
        while t < n and airborne[t]:
            t += 1
        if t - s >= MIN_FLIGHT:
            g[s:t] = -2 * np.polyfit(np.arange(t - s) / FPS, com_y[s:t], 2)[0]
    return g


def physics_labels(j):
    T = j.shape[0]
    com, contact, feet = center_of_mass(j), foot_contacts(j), j[:, FOOT_JOINTS]
    pts = [feet[t, contact[t]][:, [0, 2]] for t in range(T)]
    margin = np.array([balance_margin(com[t, [0, 2]], pts[t]) for t in range(T)])
    # Extrapolated CoM (Hof 2005): position + velocity / omega0. Unlike the static CoM test,
    # normal walking keeps it near the support, so a negative value is a real balance problem.
    v_com = np.gradient(com, 1.0 / FPS, axis=0)
    omega0 = np.sqrt(9.81 / max(np.median(com[:, 1]), 0.3))
    xcom = com[:, [0, 2]] + v_com[:, [0, 2]] / omega0
    xmargin = np.array([balance_margin(xcom[t], pts[t]) for t in range(T)])
    vel = np.zeros(feet.shape[:2])
    vel[1:] = np.linalg.norm((feet[1:] - feet[:-1])[..., [0, 2]], axis=-1) * FPS
    vel[0] = vel[1]
    skate = np.where(contact.any(1), (vel * contact).sum(1) / np.maximum(contact.sum(1), 1), np.nan)
    # Airborne = every foot joint above its height threshold (pure geometry, so sliding feet don't count)
    # and no knee, wrist or pelvis near the floor (crawling and cartwheels are supported, not in flight).
    airborne = (feet[..., 1] > FOOT_HEIGHT_MAX[None]).all(1) & (j[:, SUPPORT_JOINTS, 1] > SUPPORT_HEIGHT_MIN).all(1)
    return dict(xcom_margin=xmargin, balance_margin=margin, lowest_foot=feet[..., 1].min(1),
                in_flight=(~contact.any(1)).astype(np.float32), skate=skate, contact=contact,
                airborne=airborne.astype(np.float32), flight_gravity=flight_gravity(com[:, 1], airborne))


def labels_for(joints):
    out = [physics_labels(j) for j in joints]
    return {k: np.stack([o[k] for o in out]) for k in out[0]}


