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
