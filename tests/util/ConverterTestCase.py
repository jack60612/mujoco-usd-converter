# SPDX-FileCopyrightText: Copyright (c) 2025 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

import math

import omni.asset_validator
import usdex.core
import usdex.test
from pxr import Gf, UsdGeom


class ConverterTestCase(usdex.test.TestCase):

    defaultUpAxis = UsdGeom.Tokens.z  # noqa: N815

    def setUp(self):
        super().setUp()
        # All conversion results should be valid atomic assets
        self.validationEngine.enable_rule(omni.asset_validator.AnchoredAssetPathsChecker)
        self.validationEngine.enable_rule(omni.asset_validator.SupportedFileTypesChecker)

    def assertRotationsEquivalent(self, rot1: Gf.Rotation | Gf.Quatf | Gf.Quatd, rot2: Gf.Rotation | Gf.Quatf | Gf.Quatd, tolerance: float = 1e-6):
        """Compare physical rotations, with an angular error bound in degrees.

        Normalize native quaternion components in double before measuring the
        relative rotation. Axis-angle representations and quaternion signs are
        not unique; their individual components are not a rotation-error oracle.
        """
        first = Gf.Quatd(rot1.GetQuat() if isinstance(rot1, Gf.Rotation) else rot1).GetNormalized()
        second = Gf.Quatd(rot2.GetQuat() if isinstance(rot2, Gf.Rotation) else rot2).GetNormalized()
        difference = first * second.GetInverse()
        error = math.degrees(2 * math.atan2(difference.GetImaginary().GetLength(), abs(difference.GetReal())))
        self.assertLessEqual(error, tolerance, f"Relative rotation error {error} degrees exceeds {tolerance}")
