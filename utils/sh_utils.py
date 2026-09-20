#  Copyright 2021 The PlenOctree Authors.
#  Redistribution and use in source and binary forms, with or without
#  modification, are permitted provided that the following conditions are met:
#
#  1. Redistributions of source code must retain the above copyright notice,
#  this list of conditions and the following disclaimer.
#
#  2. Redistributions in binary form must reproduce the above copyright notice,
#  this list of conditions and the following disclaimer in the documentation
#  and/or other materials provided with the distribution.
#
#  THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
#  AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
#  IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
#  ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
#  LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
#  CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
#  SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
#  INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
#  CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
#  ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
#  POSSIBILITY OF SUCH DAMAGE.

import torch

C0 = 0.28209479177387814
C1 = 0.4886025119029199
C2 = [
    1.0925484305920792,
    -1.0925484305920792,
    0.31539156525252005,
    -1.0925484305920792,
    0.5462742152960396
]
C3 = [
    -0.5900435899266435,
    2.890611442640554,
    -0.4570457994644658,
    0.3731763325901154,
    -0.4570457994644658,
    1.445305721320277,
    -0.5900435899266435
]
C4 = [
    2.5033429417967046,
    -1.7701307697799304,
    0.9461746957575601,
    -0.6690465435572892,
    0.10578554691520431,
    -0.6690465435572892,
    0.47308734787878004,
    -1.7701307697799304,
    0.6258357354491761,
]
C5 = [ # ... strictly speaking we don't ened this much precision but I'm not gunna figure out how much we *do* need.
	-0.6563820568401701028141187663761445594,
	 8.3026492595241651159831940903807970761,
	-0.4892382994352503876840087158429572796,
	 4.7935367849733237550019011395632340416,
	-0.4529466511956969212984416582171542830,
	 0.1169503224534235964397151920902780822,
	-0.4529466511956969212984416582171542830,
	 2.3967683924866618775009505697816170208,
	-0.4892382994352503876840087158429572796,
	 2.0756623148810412789957985225951992690,
	-0.6563820568401701028141187663761445594
]


def eval_sh(deg, sh, dirs):
    """
    Evaluate spherical harmonics at unit directions
    using hardcoded SH polynomials.
    Works with torch/np/jnp.
    ... Can be 0 or more batch dimensions.
    Args:
        deg: int SH deg. Currently, 0-5 supported
        sh: jnp.ndarray SH coeffs [..., C, (deg + 1) ** 2]
        dirs: jnp.ndarray unit directions [..., 3]
    Returns:
        [..., C]
    """
    assert deg <= 5 and deg >= 0
    coeff = (deg + 1) ** 2
    assert sh.shape[-1] >= coeff

    result = C0 * sh[..., 0]
    if deg > 0:
        x, y, z = dirs[..., 0:1], dirs[..., 1:2], dirs[..., 2:3]
        result = (result -
                C1 * y * sh[..., 1] +
                C1 * z * sh[..., 2] -
                C1 * x * sh[..., 3])

        if deg > 1:
            xx, yy, zz = x * x, y * y, z * z
            xy, yz, xz = x * y, y * z, x * z
            xxpyy, xxmyy = xx + yy, xx - yy
            result = (result +
                    C2[0] * xy * sh[..., 4] +
                    C2[1] * yz * sh[..., 5] +
                    C2[2] * (2.0 * zz - xxpyy) * sh[..., 6] +
                    C2[3] * xz * sh[..., 7] +
                    C2[4] * xxmyy * sh[..., 8])

            if deg > 2:
                result = (result +
                C3[0] * y * (3 * xx - yy) * sh[..., 9] +
                C3[1] * xy * z * sh[..., 10] +
                C3[2] * y * (4 * zz - xxpyy)* sh[..., 11] +
                C3[3] * z * (2 * zz - 3 * xxpyy) * sh[..., 12] +
                C3[4] * x * (4 * zz - xxpyy) * sh[..., 13] +
                C3[5] * z * xxmyy * sh[..., 14] +
                C3[6] * x * (xx - 3 * yy) * sh[..., 15])

                # if deg > 3:
                #     result = (result + C4[0] * xy * (xx - yy) * sh[..., 16] +
                #             C4[1] * yz * (3 * xx - yy) * sh[..., 17] +
                #             C4[2] * xy * (7 * zz - 1) * sh[..., 18] + # This isn't very consistent: if they do -1 here instead of 6zz - xx - yy, they should do
                #                                                       # the same thing in C2[2] with 3zz - 1.  zz + yy + xx SHOULD be 1, but with rounding
                #                                                       # it might not be - I *think* it avoids bias to use the 2zz - xx - yy, so we should
                #                                                       # maintain that convention where possible even though it makes some of the equations
                #                                                       # considerably more involved/ugly - though it might just not matter at all.  I actually
                #             C4[3] * yz * (7 * zz - 3) * sh[..., 19] +
                #             C4[4] * (zz * (35 * zz - 30) + 3) * sh[..., 20] +
                #             C4[5] * xz * (7 * zz - 3) * sh[..., 21] +
                #             C4[6] * (xx - yy) * (7 * zz - 1) * sh[..., 22] +
                #             C4[7] * xz * (xx - 3 * yy) * sh[..., 23] +
                #             C4[8] * (xx * (xx - 3 * yy) - yy * (3 * xx - yy)) * sh[..., 24])

                if deg > 3:
                    result = (result + C4[0] * xy * xxmyy * sh[..., 16] +
                            C4[1] * yz * (3 * xx - yy) * sh[..., 17] +
                            C4[2] * xy * (6 * zz - xxpyy) * sh[..., 18] +
                            C4[3] * yz * (4 * zz - 3 * xxpyy) * sh[..., 19] +
                            C4[4] * (8 * zz * (zz - 3 * xxpyy) + 3 * xxpyy * xxpyy) * sh[..., 20] +
                            C4[5] * xz * (4 * zz - 3 * xxpyy) * sh[..., 21] +
                            C4[6] * (xx * (6 * zz - xx) + yy * (yy - 6 * zz)) * sh[..., 22] +
                            C4[7] * xz * (xx - 3 * yy) * sh[..., 23] +
                            C4[8] * (xxmyy - 2 * xy) * (xxmyy + 2 * xy) * sh[..., 24])

                    if deg > 4:
                        result = (result + C5[0] * y * (5 * xx * (xx - 2 * yy) + yy * yy) * sh[..., 25] +
                                C5[1]  * xy * z * xxmyy * sh[..., 26] +
                                C5[2]  * y * (3 * xx - yy) * (8 * zz - xxpyy) * sh[..., 27] +
                                C5[3]  * xy * z * (2 * zz - xxpyy) * sh[..., 28] +
                                C5[4]  * y * (xxpyy * xxpyy + zz * (8 * zz - 12 * xxpyy)) * sh[..., 29] +
                                C5[5]  * z * (15 * xxpyy * xxpyy + 8 * zz * (zz - 5 * xxpyy)) * sh[..., 30] +
                                C5[6]  * x * (xxpyy * xxpyy + 4 * zz * (2 * zz - 3 * xxpyy)) * sh[..., 31] +
                                C5[7]  * z * (xx * (2 * zz - xx) + yy * (yy - 2 * zz)) * sh[..., 32] +
                                C5[8]  * x * (xx - 3 * yy) * (8 * zz - xxpyy) * sh[..., 33] + 
                                C5[9]  * z * (xxmyy - 2 * xy) * (xxmyy + 2 * xy) * sh[..., 34] +
                                C5[10] * x * (5 * yy * (yy - 2 * xx) + xx * xx) * sh[..., 35])
                """
					result = result +
						// See backward.cu for a few more commments, not many
						SH_C4[0] * xy * xxmyy * sh[16] + 
						SH_C4[1] * yz * (3.0f * xx - yy) * sh[17] + 
						SH_C4[2] * xy * (6.0f * zz - xxpyy) * sh[18] + 
						SH_C4[3] * yz * (4.0f * zz - 3.0f * xxpyy) * sh[19] +
						SH_C4[4] * (8 * zz * (zz - 3.0f * xxpyy) + 3 * xxpyy * xxpyy) * sh[20] +
						SH_C4[5] * xz * (4.0f * zz - 3.0f * xxpyy) * sh[21] + 
						SH_C4[6] * (xx * (6.0f * zz - xx) + yy * (yy - 6.0f * zz)) * sh[22] +
						SH_C4[7] * xz * (xx - 3.0f * yy) * sh[23] +
						SH_C4[8] * (xxmyy - 2.0f * xy) * (xxmyy + 2.0f * xy) * sh[24];
					
					if (deg > 4)
					{
						result = result +
							SH_C5[0]  * y * (5.f * xx * (xx - 2.f * yy) + yy * yy) * sh[25] +
							SH_C5[1]  * xy * z * xxmyy * sh[26] +
							SH_C5[2]  * y * (3.f * xx - yy) * (8.f * zz - xxpyy) * sh[27] +
							SH_C5[3]  * xy * z * (2.f * zz - xxpyy) * sh[28] +
							SH_C5[4]  * y * (xxpyy * xxpyy + zz * (8.f * zz - 12.f * xxpyy)) * sh[29] +
							SH_C5[5]  * z * (15.f * xxpyy * xxpyy + 8.f * zz * (zz - 5.f * xxpyy)) * sh[30] +
							SH_C5[6]  * x * (xxpyy * xxpyy + 4.f * zz * (2.f * zz - 3.f * xxpyy)) * sh[31] +
							SH_C5[7]  * z * (xx * (2.f * zz - xx) + yy * (yy - 2.f ** zz)) * sh[32] +
							SH_C5[8]  * x * (xx - 3.f * yy) * (8.f * zz - xxpyy) * sh[33] +
							SH_C5[9]  * z * (xxmyy - 2.f * xy) * (xxmyy + 2.f * xy) * sh[34] +
							SH_C5[10] * x * (5.f * yy * (yy - 2.f * xx) + xx * xx) * sh[35];
                """


    return result

def RGB2SH(rgb):
    return (rgb - 0.5) / C0

def SH2RGB(sh):
    return sh * C0 + 0.5