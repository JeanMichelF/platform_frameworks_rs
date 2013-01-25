/*
 * Copyright (C) 2011 The Android Open Source Project
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *      http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include <stdint.h>
#include <x86intrin.h>

/* Unsigned extend packed 8-bit integer (in LBS) into packed 32-bit integer */
static inline __m128i cvtepu8_epi32(__m128i x) {
#if defined(__SSE4_1__)
    return _mm_cvtepu8_epi32(x);
#elif defined(__SSSE3__)
    const __m128i M8to32 = _mm_set_epi32(0xffffff03, 0xffffff02, 0xffffff01, 0xffffff00);
    x = _mm_shuffle_epi8(x, M8to32);
    return x;
#else
#   error "Require at least SSSE3"
#endif
}

/* Signed extend packed 16-bit integer (in LBS) into packed 32-bit integer */
static inline __m128i cvtepi16_epi32(__m128i x) {
#if defined(__SSE4_1__)
    return _mm_cvtepi16_epi32(x);
#elif defined(__SSSE3__)
    const __m128i M16to32 = _mm_set_epi32(0xffff0706, 0xffff0504, 0xffff0302, 0xffff0100);
    x = _mm_shuffle_epi8(x, M16to32);
    x = _mm_srli_epi32(x, 16);
    return _mm_srai_epi32(x, 16);
#else
#   error "Require at least SSSE3"
#endif
}

static inline __m128i packus_epi32(__m128i lo, __m128i hi) {
#if defined(__SSE4_1__)
    return _mm_packus_epi32(lo, hi);
#elif defined(__SSSE3__)
    const __m128i C0 = _mm_set_epi32(0x0000, 0x0000, 0x0000, 0x0000);
    const __m128i C1 = _mm_set_epi32(0xffff, 0xffff, 0xffff, 0xffff);
    const __m128i M32to16L = _mm_set_epi32(0xffffffff, 0xffffffff, 0x0d0c0908, 0x05040100);
    const __m128i M32to16H = _mm_set_epi32(0x0d0c0908, 0x05040100, 0xffffffff, 0xffffffff);
    lo = _mm_and_si128(lo, _mm_cmpgt_epi32(C0, lo));
    lo = _mm_or_si128(lo, _mm_cmpgt_epi32(C1, lo));
    hi = _mm_and_si128(lo, _mm_cmpgt_epi32(C0, hi));
    hi = _mm_or_si128(lo, _mm_cmpgt_epi32(C1, hi));
    return _mm_or_si128(_mm_shuffle_epi8(lo, M32to16L),
                        _mm_shuffle_epi8(hi, M32to16H));
#else
#   error "Require at least SSSE3"
#endif
}

static inline __m128i mullo_epi32(__m128i x, __m128i y) {
#if defined(__SSE4_1__)
    return _mm_mullo_epi32(x, y);
#elif defined(__SSSE3__)
    const __m128i Meven = _mm_set_epi32(0x00000000, 0xffffffff, 0x00000000, 0xffffffff);
    __m128i even = _mm_mul_epu32(x, y);
    __m128i odd = _mm_mul_epu32(_mm_srli_si128(x, 4),
                                _mm_srli_si128(y, 4));
    even = _mm_and_si128(even, Meven);
    odd = _mm_and_si128(odd, Meven);
    return _mm_or_si128(even, _mm_slli_si128(odd, 4));
#else
#   error "Require at least SSSE3"
#endif
}

void rsdIntrinsicConvolve3x3_K(void *dst,
                               const void *y0, const void *y1, const void *y2,
                               const short *coef, uint32_t count) {
    const __m128i M0 = _mm_set_epi32(0xffffff03, 0xffffff02, 0xffffff01, 0xffffff00);
    const __m128i M1 = _mm_set_epi32(0xffffff07, 0xffffff06, 0xffffff05, 0xffffff04);
    const __m128i M2 = _mm_set_epi32(0xffffff0b, 0xffffff0a, 0xffffff09, 0xffffff08);
    const __m128i M3 = _mm_set_epi32(0xffffff0f, 0xffffff0e, 0xffffff0d, 0xffffff0c);
    __m128i x;
    __m128i c0, c1, c2, c3, c4, c5, c6, c7, c8;
    __m128i r0, r1, r2;
    __m128i p0, p1, p2, p3, p4, p5, p6, p7, p8, p9, p10, p11;
    __m128i o0, o1;
    uint32_t i;

    /* Sign-extend coefficient matrix to i32 */
    x = cvtepi16_epi32(_mm_set1_epi64(*(const __m64 *)(coef+0)));
    c0 = _mm_shuffle_epi32(x, 0x00);
    c1 = _mm_shuffle_epi32(x, 0x55);
    c2 = _mm_shuffle_epi32(x, 0xaa);
    c3 = _mm_shuffle_epi32(x, 0xff);
    x = cvtepi16_epi32(_mm_set1_epi64(*(const __m64 *)(coef+4)));
    c4 = _mm_shuffle_epi32(x, 0x00);
    c5 = _mm_shuffle_epi32(x, 0x55);
    c6 = _mm_shuffle_epi32(x, 0xaa);
    c7 = _mm_shuffle_epi32(x, 0xff);
    x = cvtepi16_epi32(_mm_set1_epi64(*(const __m64 *)(coef+8)));
    c8 = _mm_shuffle_epi32(x, 0x00);

    for (i = 0; i < count; ++i) {
        r0 = _mm_loadu_si128((const __m128i *)y0);
        r1 = _mm_loadu_si128((const __m128i *)y1);
        r2 = _mm_loadu_si128((const __m128i *)y2);

        p0  = _mm_shuffle_epi8(r0, M0);
        p1  = _mm_shuffle_epi8(r0, M1);
        p2  = _mm_shuffle_epi8(r0, M2);
        p3  = _mm_shuffle_epi8(r0, M3);

        p4  = _mm_shuffle_epi8(r1, M0);
        p5  = _mm_shuffle_epi8(r1, M1);
        p6  = _mm_shuffle_epi8(r1, M2);
        p7  = _mm_shuffle_epi8(r1, M3);

        p8  = _mm_shuffle_epi8(r2, M0);
        p9  = _mm_shuffle_epi8(r2, M1);
        p10 = _mm_shuffle_epi8(r2, M2);
        p11 = _mm_shuffle_epi8(r2, M3);

        o0 =                   mullo_epi32( p0, c0);
        o0 = _mm_add_epi32(o0, mullo_epi32( p1, c1));
        o0 = _mm_add_epi32(o0, mullo_epi32( p2, c2));
        o0 = _mm_add_epi32(o0, mullo_epi32( p4, c3));
        o0 = _mm_add_epi32(o0, mullo_epi32( p5, c4));
        o0 = _mm_add_epi32(o0, mullo_epi32( p6, c5));
        o0 = _mm_add_epi32(o0, mullo_epi32( p8, c6));
        o0 = _mm_add_epi32(o0, mullo_epi32( p9, c7));
        o0 = _mm_add_epi32(o0, mullo_epi32(p10, c8));
        o0 = _mm_srai_epi32(o0, 8);

        o1 =                   mullo_epi32( p1, c0);
        o1 = _mm_add_epi32(o1, mullo_epi32( p2, c1));
        o1 = _mm_add_epi32(o1, mullo_epi32( p3, c2));
        o1 = _mm_add_epi32(o1, mullo_epi32( p5, c3));
        o1 = _mm_add_epi32(o1, mullo_epi32( p6, c4));
        o1 = _mm_add_epi32(o1, mullo_epi32( p7, c5));
        o1 = _mm_add_epi32(o1, mullo_epi32( p9, c6));
        o1 = _mm_add_epi32(o1, mullo_epi32(p10, c7));
        o1 = _mm_add_epi32(o1, mullo_epi32(p11, c8));
        o1 = _mm_srai_epi32(o1, 8);

        o0 = packus_epi32(o0, o1);
        o0 = _mm_packus_epi16(o0, o0);
        _mm_storel_epi64((__m128i *)dst, o0);

        y0 = (const char *)y0 + 8;
        y1 = (const char *)y1 + 8;
        y2 = (const char *)y2 + 8;
        dst = (char *)dst + 8;
    }
}

void rsdIntrinsicColorMatrix4x4_K(void *dst, const void *src,
                                  const short *coef, uint32_t count) {
    const __m128i T4x4 = _mm_set_epi8(15, 11, 7, 3,
                                      14, 10, 6, 2,
                                      13,  9, 5, 1,
                                      12,  8, 4, 0);
    const __m128i Mx = _mm_set_epi32(0xffffff0c, 0xffffff08, 0xffffff04, 0xffffff00);
    const __m128i My = _mm_set_epi32(0xffffff0d, 0xffffff09, 0xffffff05, 0xffffff01);
    const __m128i Mz = _mm_set_epi32(0xffffff0e, 0xffffff0a, 0xffffff06, 0xffffff02);
    const __m128i Mw = _mm_set_epi32(0xffffff0f, 0xffffff0b, 0xffffff07, 0xffffff03);
    __m128i c0, c1, c2, c3;
    __m128i i4, o4;
    __m128i x, y, z, w;
    __m128i x2, y2, z2, w2;
    uint32_t i;

    /* Sign-extend coefficient matrix to i32 */
    c0 = cvtepi16_epi32(_mm_set1_epi64(*(const __m64 *)(coef+0)));
    c1 = cvtepi16_epi32(_mm_set1_epi64(*(const __m64 *)(coef+4)));
    c2 = cvtepi16_epi32(_mm_set1_epi64(*(const __m64 *)(coef+8)));
    c3 = cvtepi16_epi32(_mm_set1_epi64(*(const __m64 *)(coef+12)));

    for (i = 0; i < count; ++i) {
        i4 = _mm_load_si128((const __m128i *)src);

        x = _mm_shuffle_epi8(i4, Mx);
        y = _mm_shuffle_epi8(i4, My);
        z = _mm_shuffle_epi8(i4, Mz);
        w = _mm_shuffle_epi8(i4, Mw);

        x2 =                   mullo_epi32(x, _mm_shuffle_epi32(c0, 0x00));
        y2 =                   mullo_epi32(x, _mm_shuffle_epi32(c0, 0x55));
        z2 =                   mullo_epi32(x, _mm_shuffle_epi32(c0, 0xaa));
        w2 =                   mullo_epi32(x, _mm_shuffle_epi32(c0, 0xff));

        x2 = _mm_add_epi32(x2, mullo_epi32(y, _mm_shuffle_epi32(c1, 0x00)));
        y2 = _mm_add_epi32(y2, mullo_epi32(y, _mm_shuffle_epi32(c1, 0x55)));
        z2 = _mm_add_epi32(z2, mullo_epi32(y, _mm_shuffle_epi32(c1, 0xaa)));
        w2 = _mm_add_epi32(w2, mullo_epi32(y, _mm_shuffle_epi32(c1, 0xff)));

        x2 = _mm_add_epi32(x2, mullo_epi32(z, _mm_shuffle_epi32(c2, 0x00)));
        y2 = _mm_add_epi32(y2, mullo_epi32(z, _mm_shuffle_epi32(c2, 0x55)));
        z2 = _mm_add_epi32(z2, mullo_epi32(z, _mm_shuffle_epi32(c2, 0xaa)));
        w2 = _mm_add_epi32(w2, mullo_epi32(z, _mm_shuffle_epi32(c2, 0xff)));

        x2 = _mm_add_epi32(x2, mullo_epi32(w, _mm_shuffle_epi32(c3, 0x00)));
        y2 = _mm_add_epi32(y2, mullo_epi32(w, _mm_shuffle_epi32(c3, 0x55)));
        z2 = _mm_add_epi32(z2, mullo_epi32(w, _mm_shuffle_epi32(c3, 0xaa)));
        w2 = _mm_add_epi32(w2, mullo_epi32(w, _mm_shuffle_epi32(c3, 0xff)));

        x2 = _mm_srai_epi32(x2, 8);
        y2 = _mm_srai_epi32(y2, 8);
        z2 = _mm_srai_epi32(z2, 8);
        w2 = _mm_srai_epi32(w2, 8);

        x2 = packus_epi32(x2, y2);
        z2 = packus_epi32(z2, w2);
        o4 = _mm_packus_epi16(x2, z2);

        o4 = _mm_shuffle_epi8(o4, T4x4);
        _mm_store_si128((__m128i *)dst, o4);

        src = (const char *)src + 16;
        dst = (char *)dst + 16;
    }
}

void rsdIntrinsicColorMatrix3x3_K(void *dst, const void *src,
                                  const short *coef, uint32_t count) {
    const __m128i T4x4 = _mm_set_epi8(15, 11, 7, 3,
                                      14, 10, 6, 2,
                                      13,  9, 5, 1,
                                      12,  8, 4, 0);
    const __m128i Mx = _mm_set_epi32(0xffffff0c, 0xffffff08, 0xffffff04, 0xffffff00);
    const __m128i My = _mm_set_epi32(0xffffff0d, 0xffffff09, 0xffffff05, 0xffffff01);
    const __m128i Mz = _mm_set_epi32(0xffffff0e, 0xffffff0a, 0xffffff06, 0xffffff02);
    const __m128i Mw = _mm_set_epi32(0xffffff0f, 0xffffff0b, 0xffffff07, 0xffffff03);
    __m128i c0, c1, c2, c3;
    __m128i i4, o4;
    __m128i x, y, z, w;
    __m128i x2, y2, z2, w2;
    uint32_t i;

    /* Sign-extend coefficient matrix to i32 */
    c0 = cvtepi16_epi32(_mm_set1_epi64(*(const __m64 *)(coef+0)));
    c1 = cvtepi16_epi32(_mm_set1_epi64(*(const __m64 *)(coef+4)));
    c2 = cvtepi16_epi32(_mm_set1_epi64(*(const __m64 *)(coef+8)));
    c3 = cvtepi16_epi32(_mm_set1_epi64(*(const __m64 *)(coef+12)));

    for (i = 0; i < count; ++i) {
        i4 = _mm_load_si128((const __m128i *)src);

        x = _mm_shuffle_epi8(i4, Mx);
        y = _mm_shuffle_epi8(i4, My);
        z = _mm_shuffle_epi8(i4, Mz);
        w = _mm_shuffle_epi8(i4, Mw);

        x2 =                   mullo_epi32(x, _mm_shuffle_epi32(c0, 0x00));
        y2 =                   mullo_epi32(x, _mm_shuffle_epi32(c0, 0x55));
        z2 =                   mullo_epi32(x, _mm_shuffle_epi32(c0, 0xaa));

        x2 = _mm_add_epi32(x2, mullo_epi32(y, _mm_shuffle_epi32(c1, 0x00)));
        y2 = _mm_add_epi32(y2, mullo_epi32(y, _mm_shuffle_epi32(c1, 0x55)));
        z2 = _mm_add_epi32(z2, mullo_epi32(y, _mm_shuffle_epi32(c1, 0xaa)));

        x2 = _mm_add_epi32(x2, mullo_epi32(z, _mm_shuffle_epi32(c2, 0x00)));
        y2 = _mm_add_epi32(y2, mullo_epi32(z, _mm_shuffle_epi32(c2, 0x55)));
        z2 = _mm_add_epi32(z2, mullo_epi32(z, _mm_shuffle_epi32(c2, 0xaa)));

        x2 = _mm_add_epi32(x2, mullo_epi32(w, _mm_shuffle_epi32(c3, 0x00)));
        y2 = _mm_add_epi32(y2, mullo_epi32(w, _mm_shuffle_epi32(c3, 0x55)));
        z2 = _mm_add_epi32(z2, mullo_epi32(w, _mm_shuffle_epi32(c3, 0xaa)));

        x2 = _mm_srai_epi32(x2, 8);
        y2 = _mm_srai_epi32(y2, 8);
        z2 = _mm_srai_epi32(z2, 8);
        w2 = w;

        x2 = packus_epi32(x2, y2);
        z2 = packus_epi32(z2, w2);
        o4 = _mm_packus_epi16(x2, z2);

        o4 = _mm_shuffle_epi8(o4, T4x4);
        _mm_store_si128((__m128i *)dst, o4);

        src = (const char *)src + 16;
        dst = (char *)dst + 16;
    }
}

void rsdIntrinsicColorMatrixDot_K(void *dst, const void *src,
                                  const short *coef, uint32_t count) {
    const __m128i T4x4 = _mm_set_epi8(15, 11, 7, 3,
                                      14, 10, 6, 2,
                                      13,  9, 5, 1,
                                      12,  8, 4, 0);
    const __m128i Mx = _mm_set_epi32(0xffffff0c, 0xffffff08, 0xffffff04, 0xffffff00);
    const __m128i My = _mm_set_epi32(0xffffff0d, 0xffffff09, 0xffffff05, 0xffffff01);
    const __m128i Mz = _mm_set_epi32(0xffffff0e, 0xffffff0a, 0xffffff06, 0xffffff02);
    const __m128i Mw = _mm_set_epi32(0xffffff0f, 0xffffff0b, 0xffffff07, 0xffffff03);
    __m128i c0, c1, c2, c3;
    __m128i i4, o4;
    __m128i x, y, z, w;
    __m128i x2, y2, z2, w2;
    uint32_t i;

    /* Sign-extend coefficient matrix to i32 */
    c0 = cvtepi16_epi32(_mm_set1_epi64(*(const __m64 *)(coef+0)));
    c1 = cvtepi16_epi32(_mm_set1_epi64(*(const __m64 *)(coef+4)));
    c2 = cvtepi16_epi32(_mm_set1_epi64(*(const __m64 *)(coef+8)));
    c3 = cvtepi16_epi32(_mm_set1_epi64(*(const __m64 *)(coef+12)));

    for (i = 0; i < count; ++i) {
        i4 = _mm_load_si128((const __m128i *)src);

        x = _mm_shuffle_epi8(i4, Mx);
        y = _mm_shuffle_epi8(i4, My);
        z = _mm_shuffle_epi8(i4, Mz);
        w = _mm_shuffle_epi8(i4, Mw);

        x2 =                   mullo_epi32(x, _mm_shuffle_epi32(c0, 0x00));
        x2 = _mm_add_epi32(x2, mullo_epi32(y, _mm_shuffle_epi32(c1, 0x00)));
        x2 = _mm_add_epi32(x2, mullo_epi32(z, _mm_shuffle_epi32(c2, 0x00)));
        x2 = _mm_add_epi32(x2, mullo_epi32(w, _mm_shuffle_epi32(c3, 0x00)));

        x2 = _mm_srai_epi32(x2, 8);
        y2 = x2;
        z2 = x2;
        w2 = w;

        x2 = packus_epi32(x2, y2);
        z2 = packus_epi32(z2, w2);
        o4 = _mm_packus_epi16(x2, z2);

        o4 = _mm_shuffle_epi8(o4, T4x4);
        _mm_store_si128((__m128i *)dst, o4);

        src = (const char *)src + 16;
        dst = (char *)dst + 16;
    }
}

void rsdIntrinsicBlurVF_K(void *dst,
                          const void *pin, int stride, const void *gptr,
                          int rct, int x1, int x2) {
    const char *pi;
    __m128i pi0, pi1;
    __m128 pf0, pf1;
    __m128 bp0, bp1;
    __m128 x;
    int r;

    for (; x1 < x2; x1 += 2) {
        pi = (const char *)pin + (x1 << 2);
        bp0 = _mm_setzero_ps();
        bp1 = _mm_setzero_ps();

        for (r = 0; r < rct; ++r) {
            x = _mm_load_ss((const float *)gptr + r);
            x = _mm_shuffle_ps(x, x, _MM_SHUFFLE(0, 0, 0, 0));

            pi0 = _mm_cvtsi32_si128(*(const int *)pi);
            pi1 = _mm_cvtsi32_si128(*((const int *)pi + 1));

            pf0 = _mm_cvtepi32_ps(cvtepu8_epi32(pi0));
            pf1 = _mm_cvtepi32_ps(cvtepu8_epi32(pi1));

            bp0 = _mm_add_ps(bp0, _mm_mul_ps(pf0, x));
            bp1 = _mm_add_ps(bp1, _mm_mul_ps(pf1, x));

            pi += stride;
        }

        _mm_store_ps((float *)dst, bp0);
        _mm_store_ps((float *)dst + 4, bp1);
        dst = (char *)dst + 32;
    }
}

void rsdIntrinsicBlurHF_K(void *dst,
                          const void *pin, const void *gptr,
                          int rct, int x1, int x2) {
    const __m128i Mu8 = _mm_set_epi32(0xffffffff, 0xffffffff, 0xffffffff, 0x0c080400);
    const float *pi;
    __m128 pf, x, y;
    __m128i o;
    int r;

    for (; x1 < x2; ++x1) {
        /* rct is define as 2*r+1 by the caller */
        x = _mm_load_ss((const float *)gptr);
        x = _mm_shuffle_ps(x, x, _MM_SHUFFLE(0, 0, 0, 0));

        pi = (const float *)pin + (x1 << 2);
        pf = _mm_mul_ps(x, _mm_load_ps(pi));

        for (r = 1; r < rct; r += 2) {
            x = _mm_load_ss((const float *)gptr + r);
            y = _mm_load_ss((const float *)gptr + r + 1);
            x = _mm_shuffle_ps(x, x, _MM_SHUFFLE(0, 0, 0, 0));
            y = _mm_shuffle_ps(y, y, _MM_SHUFFLE(0, 0, 0, 0));

            pf = _mm_add_ps(pf, _mm_mul_ps(x, _mm_load_ps(pi + (r << 2))));
            pf = _mm_add_ps(pf, _mm_mul_ps(y, _mm_load_ps(pi + (r << 2) + 4)));
        }

        o = _mm_cvtps_epi32(pf);
        *(int *)dst = _mm_cvtsi128_si32(_mm_shuffle_epi8(o, Mu8));
        dst = (char *)dst + 4;
    }
}

void rsdIntrinsicConvolve5x5_K(void *dst, const void *y0, const void *y1,
                               const void *y2, const void *y3, const void *y4,
                               const short *coef, uint32_t count) {
    const __m128i M0 = _mm_set_epi32(0xffffff03, 0xffffff02, 0xffffff01, 0xffffff00);
    const __m128i M1 = _mm_set_epi32(0xffffff07, 0xffffff06, 0xffffff05, 0xffffff04);
    const __m128i M2 = _mm_set_epi32(0xffffff0b, 0xffffff0a, 0xffffff09, 0xffffff08);
    const __m128i M3 = _mm_set_epi32(0xffffff0f, 0xffffff0e, 0xffffff0d, 0xffffff0c);
    __m128i x;
    __m128i  c0,  c1,  c2,  c3,  c4;
    __m128i  c5,  c6,  c7,  c8,  c9;
    __m128i c10, c11, c12, c13, c14;
    __m128i c15, c16, c17, c18, c19;
    __m128i c20, c21, c22, c23, c24;
    __m128i r0, r1, r2, r3, r4, r5, r6, r7, r8, r9;
    __m128i  p0,  p1,  p2,  p3,  p4,  p5,  p6,  p7;
    __m128i  p8,  p9, p10, p11, p12, p13, p14, p15;
    __m128i p16, p17, p18, p19, p20, p21, p22, p23;
    __m128i p24, p25, p26, p27, p28, p29, p30, p31;
    __m128i p32, p33, p34, p35, p36, p37, p38, p39;
    __m128i o0, o1, o2, o3;
    uint32_t i;

    /* Sign-extend coefficient matrix to i32 */
    x = cvtepi16_epi32(_mm_set1_epi64(*(const __m64 *)(coef+0)));
    c0  = _mm_shuffle_epi32(x, 0x00);
    c1  = _mm_shuffle_epi32(x, 0x55);
    c2  = _mm_shuffle_epi32(x, 0xaa);
    c3  = _mm_shuffle_epi32(x, 0xff);
    x = cvtepi16_epi32(_mm_set1_epi64(*(const __m64 *)(coef+4)));
    c4  = _mm_shuffle_epi32(x, 0x00);
    c5  = _mm_shuffle_epi32(x, 0x55);
    c6  = _mm_shuffle_epi32(x, 0xaa);
    c7  = _mm_shuffle_epi32(x, 0xff);
    x = cvtepi16_epi32(_mm_set1_epi64(*(const __m64 *)(coef+8)));
    c8  = _mm_shuffle_epi32(x, 0x00);
    c9  = _mm_shuffle_epi32(x, 0x55);
    c10 = _mm_shuffle_epi32(x, 0xaa);
    c11 = _mm_shuffle_epi32(x, 0xff);
    x = cvtepi16_epi32(_mm_set1_epi64(*(const __m64 *)(coef+12)));
    c12 = _mm_shuffle_epi32(x, 0x00);
    c13 = _mm_shuffle_epi32(x, 0x55);
    c14 = _mm_shuffle_epi32(x, 0xaa);
    c15 = _mm_shuffle_epi32(x, 0xff);
    x = cvtepi16_epi32(_mm_set1_epi64(*(const __m64 *)(coef+16)));
    c16 = _mm_shuffle_epi32(x, 0x00);
    c17 = _mm_shuffle_epi32(x, 0x55);
    c18 = _mm_shuffle_epi32(x, 0xaa);
    c19 = _mm_shuffle_epi32(x, 0xff);
    x = cvtepi16_epi32(_mm_set1_epi64(*(const __m64 *)(coef+20)));
    c20 = _mm_shuffle_epi32(x, 0x00);
    c21 = _mm_shuffle_epi32(x, 0x55);
    c22 = _mm_shuffle_epi32(x, 0xaa);
    c23 = _mm_shuffle_epi32(x, 0xff);
    x = cvtepi16_epi32(_mm_set1_epi64(*(const __m64 *)(coef+24)));
    c24 = _mm_shuffle_epi32(x, 0x00);

    for (i = 0; i < count; ++i) {
        r0 = _mm_loadu_si128((const __m128i *)y0);
        r1 = _mm_loadu_si128((const __m128i *)y0 + 1);
        r2 = _mm_loadu_si128((const __m128i *)y1);
        r3 = _mm_loadu_si128((const __m128i *)y1 + 1);
        r4 = _mm_loadu_si128((const __m128i *)y2);
        r5 = _mm_loadu_si128((const __m128i *)y2 + 1);
        r6 = _mm_loadu_si128((const __m128i *)y3);
        r7 = _mm_loadu_si128((const __m128i *)y3 + 1);
        r8 = _mm_loadu_si128((const __m128i *)y4);
        r9 = _mm_loadu_si128((const __m128i *)y4 + 1);

        p0  = _mm_shuffle_epi8(r0, M0);
        p1  = _mm_shuffle_epi8(r0, M1);
        p2  = _mm_shuffle_epi8(r0, M2);
        p3  = _mm_shuffle_epi8(r0, M3);
        p4  = _mm_shuffle_epi8(r1, M0);
        p5  = _mm_shuffle_epi8(r1, M1);
        p6  = _mm_shuffle_epi8(r1, M2);
        p7  = _mm_shuffle_epi8(r1, M3);

        p8  = _mm_shuffle_epi8(r2, M0);
        p9  = _mm_shuffle_epi8(r2, M1);
        p10 = _mm_shuffle_epi8(r2, M2);
        p11 = _mm_shuffle_epi8(r2, M3);
        p12 = _mm_shuffle_epi8(r3, M0);
        p13 = _mm_shuffle_epi8(r3, M1);
        p14 = _mm_shuffle_epi8(r3, M2);
        p15 = _mm_shuffle_epi8(r3, M3);

        p16 = _mm_shuffle_epi8(r4, M0);
        p17 = _mm_shuffle_epi8(r4, M1);
        p18 = _mm_shuffle_epi8(r4, M2);
        p19 = _mm_shuffle_epi8(r4, M3);
        p20 = _mm_shuffle_epi8(r5, M0);
        p21 = _mm_shuffle_epi8(r5, M1);
        p22 = _mm_shuffle_epi8(r5, M2);
        p23 = _mm_shuffle_epi8(r5, M3);

        p24 = _mm_shuffle_epi8(r6, M0);
        p25 = _mm_shuffle_epi8(r6, M1);
        p26 = _mm_shuffle_epi8(r6, M2);
        p27 = _mm_shuffle_epi8(r6, M3);
        p28 = _mm_shuffle_epi8(r7, M0);
        p29 = _mm_shuffle_epi8(r7, M1);
        p30 = _mm_shuffle_epi8(r7, M2);
        p31 = _mm_shuffle_epi8(r7, M3);

        p32 = _mm_shuffle_epi8(r8, M0);
        p33 = _mm_shuffle_epi8(r8, M1);
        p34 = _mm_shuffle_epi8(r8, M2);
        p35 = _mm_shuffle_epi8(r8, M3);
        p36 = _mm_shuffle_epi8(r9, M0);
        p37 = _mm_shuffle_epi8(r9, M1);
        p38 = _mm_shuffle_epi8(r9, M2);
        p39 = _mm_shuffle_epi8(r9, M3);

        o0 =                   mullo_epi32( p0,  c0);
        o0 = _mm_add_epi32(o0, mullo_epi32( p1,  c1));
        o0 = _mm_add_epi32(o0, mullo_epi32( p2,  c2));
        o0 = _mm_add_epi32(o0, mullo_epi32( p3,  c3));
        o0 = _mm_add_epi32(o0, mullo_epi32( p4,  c4));
        o0 = _mm_add_epi32(o0, mullo_epi32( p8,  c5));
        o0 = _mm_add_epi32(o0, mullo_epi32( p9,  c6));
        o0 = _mm_add_epi32(o0, mullo_epi32(p10,  c7));
        o0 = _mm_add_epi32(o0, mullo_epi32(p11,  c8));
        o0 = _mm_add_epi32(o0, mullo_epi32(p12,  c9));
        o0 = _mm_add_epi32(o0, mullo_epi32(p16, c10));
        o0 = _mm_add_epi32(o0, mullo_epi32(p17, c11));
        o0 = _mm_add_epi32(o0, mullo_epi32(p18, c12));
        o0 = _mm_add_epi32(o0, mullo_epi32(p19, c13));
        o0 = _mm_add_epi32(o0, mullo_epi32(p20, c14));
        o0 = _mm_add_epi32(o0, mullo_epi32(p24, c15));
        o0 = _mm_add_epi32(o0, mullo_epi32(p25, c16));
        o0 = _mm_add_epi32(o0, mullo_epi32(p26, c17));
        o0 = _mm_add_epi32(o0, mullo_epi32(p27, c18));
        o0 = _mm_add_epi32(o0, mullo_epi32(p28, c19));
        o0 = _mm_add_epi32(o0, mullo_epi32(p32, c20));
        o0 = _mm_add_epi32(o0, mullo_epi32(p33, c21));
        o0 = _mm_add_epi32(o0, mullo_epi32(p34, c22));
        o0 = _mm_add_epi32(o0, mullo_epi32(p35, c23));
        o0 = _mm_add_epi32(o0, mullo_epi32(p36, c24));
        o0 = _mm_srai_epi32(o0, 8);

        o1 =                   mullo_epi32( p1,  c0);
        o1 = _mm_add_epi32(o1, mullo_epi32( p2,  c1));
        o1 = _mm_add_epi32(o1, mullo_epi32( p3,  c2));
        o1 = _mm_add_epi32(o1, mullo_epi32( p4,  c3));
        o1 = _mm_add_epi32(o1, mullo_epi32( p5,  c4));
        o1 = _mm_add_epi32(o1, mullo_epi32( p9,  c5));
        o1 = _mm_add_epi32(o1, mullo_epi32(p10,  c6));
        o1 = _mm_add_epi32(o1, mullo_epi32(p11,  c7));
        o1 = _mm_add_epi32(o1, mullo_epi32(p12,  c8));
        o1 = _mm_add_epi32(o1, mullo_epi32(p13,  c9));
        o1 = _mm_add_epi32(o1, mullo_epi32(p17, c10));
        o1 = _mm_add_epi32(o1, mullo_epi32(p18, c11));
        o1 = _mm_add_epi32(o1, mullo_epi32(p19, c12));
        o1 = _mm_add_epi32(o1, mullo_epi32(p20, c13));
        o1 = _mm_add_epi32(o1, mullo_epi32(p21, c14));
        o1 = _mm_add_epi32(o1, mullo_epi32(p25, c15));
        o1 = _mm_add_epi32(o1, mullo_epi32(p26, c16));
        o1 = _mm_add_epi32(o1, mullo_epi32(p27, c17));
        o1 = _mm_add_epi32(o1, mullo_epi32(p28, c18));
        o1 = _mm_add_epi32(o1, mullo_epi32(p29, c19));
        o1 = _mm_add_epi32(o1, mullo_epi32(p33, c20));
        o1 = _mm_add_epi32(o1, mullo_epi32(p34, c21));
        o1 = _mm_add_epi32(o1, mullo_epi32(p35, c22));
        o1 = _mm_add_epi32(o1, mullo_epi32(p36, c23));
        o1 = _mm_add_epi32(o1, mullo_epi32(p37, c24));
        o1 = _mm_srai_epi32(o1, 8);

        o2 =                   mullo_epi32( p2,  c0);
        o2 = _mm_add_epi32(o2, mullo_epi32( p3,  c1));
        o2 = _mm_add_epi32(o2, mullo_epi32( p4,  c2));
        o2 = _mm_add_epi32(o2, mullo_epi32( p5,  c3));
        o2 = _mm_add_epi32(o2, mullo_epi32( p6,  c4));
        o2 = _mm_add_epi32(o2, mullo_epi32(p10,  c5));
        o2 = _mm_add_epi32(o2, mullo_epi32(p11,  c6));
        o2 = _mm_add_epi32(o2, mullo_epi32(p12,  c7));
        o2 = _mm_add_epi32(o2, mullo_epi32(p13,  c8));
        o2 = _mm_add_epi32(o2, mullo_epi32(p14,  c9));
        o2 = _mm_add_epi32(o2, mullo_epi32(p18, c10));
        o2 = _mm_add_epi32(o2, mullo_epi32(p19, c11));
        o2 = _mm_add_epi32(o2, mullo_epi32(p20, c12));
        o2 = _mm_add_epi32(o2, mullo_epi32(p21, c13));
        o2 = _mm_add_epi32(o2, mullo_epi32(p22, c14));
        o2 = _mm_add_epi32(o2, mullo_epi32(p26, c15));
        o2 = _mm_add_epi32(o2, mullo_epi32(p27, c16));
        o2 = _mm_add_epi32(o2, mullo_epi32(p28, c17));
        o2 = _mm_add_epi32(o2, mullo_epi32(p29, c18));
        o2 = _mm_add_epi32(o2, mullo_epi32(p30, c19));
        o2 = _mm_add_epi32(o2, mullo_epi32(p34, c20));
        o2 = _mm_add_epi32(o2, mullo_epi32(p35, c21));
        o2 = _mm_add_epi32(o2, mullo_epi32(p36, c22));
        o2 = _mm_add_epi32(o2, mullo_epi32(p37, c23));
        o2 = _mm_add_epi32(o2, mullo_epi32(p38, c24));
        o2 = _mm_srai_epi32(o2, 8);

        o3 =                   mullo_epi32( p3,  c0);
        o3 = _mm_add_epi32(o3, mullo_epi32( p4,  c1));
        o3 = _mm_add_epi32(o3, mullo_epi32( p5,  c2));
        o3 = _mm_add_epi32(o3, mullo_epi32( p6,  c3));
        o3 = _mm_add_epi32(o3, mullo_epi32( p7,  c4));
        o3 = _mm_add_epi32(o3, mullo_epi32(p11,  c5));
        o3 = _mm_add_epi32(o3, mullo_epi32(p12,  c6));
        o3 = _mm_add_epi32(o3, mullo_epi32(p13,  c7));
        o3 = _mm_add_epi32(o3, mullo_epi32(p14,  c8));
        o3 = _mm_add_epi32(o3, mullo_epi32(p15,  c9));
        o3 = _mm_add_epi32(o3, mullo_epi32(p19, c10));
        o3 = _mm_add_epi32(o3, mullo_epi32(p20, c11));
        o3 = _mm_add_epi32(o3, mullo_epi32(p21, c12));
        o3 = _mm_add_epi32(o3, mullo_epi32(p22, c13));
        o3 = _mm_add_epi32(o3, mullo_epi32(p23, c14));
        o3 = _mm_add_epi32(o3, mullo_epi32(p27, c15));
        o3 = _mm_add_epi32(o3, mullo_epi32(p28, c16));
        o3 = _mm_add_epi32(o3, mullo_epi32(p29, c17));
        o3 = _mm_add_epi32(o3, mullo_epi32(p30, c18));
        o3 = _mm_add_epi32(o3, mullo_epi32(p31, c19));
        o3 = _mm_add_epi32(o3, mullo_epi32(p35, c20));
        o3 = _mm_add_epi32(o3, mullo_epi32(p36, c21));
        o3 = _mm_add_epi32(o3, mullo_epi32(p37, c22));
        o3 = _mm_add_epi32(o3, mullo_epi32(p38, c23));
        o3 = _mm_add_epi32(o3, mullo_epi32(p39, c24));
        o3 = _mm_srai_epi32(o3, 8);

        o0 = packus_epi32(o0, o1);
        o2 = packus_epi32(o2, o3);
        o0 = _mm_packus_epi16(o0, o2);
        _mm_storeu_si128((__m128i *)dst, o0);

        y0 = (const char *)y0 + 16;
        y1 = (const char *)y1 + 16;
        y2 = (const char *)y2 + 16;
        y3 = (const char *)y3 + 16;
        y4 = (const char *)y4 + 16;
        dst = (char *)dst + 16;
    }
}
