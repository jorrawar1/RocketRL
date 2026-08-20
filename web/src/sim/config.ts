/* Constants ported from rocketenv/config.py and
 * rocketenv/sample_return/config.py. These are the values the fixture was
 * recorded with; the fixture itself carries the terrain, pads and payload. */

export const SIM = {
  dt: 1 / 60,
  g: 9.81,

  // Vehicle. T_max is rated from *dry* mass; gravity and acceleration use
  // total mass, so a loaded vehicle is genuinely harder to fly.
  dryMass: 1.0,
  dryInertia: (1.0 * 4.0 * 4.0) / 12.0,
  bodyHeight: 4.0,
  twr: 2.3,
  phiMax: (10 * Math.PI) / 180,
  thrustMultiplier: 1.0,

  fuel0: 100.0,
  burnRate: 1.65,

  padHalfW: 5.0,
  legHalfW: 1.5,
  landVyMax: 5.0,
  landVxMax: 3.5,

  worldW: 220.0,
  worldH: 140.0,
  maxSteps: 12_000,

  vRef: 20.0,
  omegaRef: 5.0,
  rayMaxRange: 80.0,
  rayAngleLo: (-150 * Math.PI) / 180,
  rayAngleHi: (-30 * Math.PI) / 180,
  rayCount: 5,

  samplingSteps: 90,
  contactArmClearance: 0.9,
  departureRadius: 7.0,

  // Reserved randomisation axes, all no-ops at the recorded settings.
  dragCoeff: 0.0,
  windX: 0.0,
} as const;

export const T_MAX = SIM.twr * SIM.dryMass * SIM.g;
