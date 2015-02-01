/***************************************************************************
 *             __________               __   ___.
 *   Open      \______   \ ____   ____ |  | _\_ |__   _______  ___
 *   Source     |       _//  _ \_/ ___\|  |/ /| __ \ /  _ \  \/  /
 *   Jukebox    |    |   (  <_> )  \___|    < | \_\ (  <_> > <  <
 *   Firmware   |____|_  /\____/ \___  >__|_ \|___  /\____/__/\_ \
 *                     \/            \/     \/    \/            \/
 * $Id: $
 *
 * Copyright (C) 2005 Dave Chapman
 *
 * All files in this archive are subject to the GNU General Public License.
 * See the file COPYING in the source tree root for full license agreement.
 *
 * This software is distributed on an "AS IS" basis, WITHOUT WARRANTY OF ANY
 * KIND, either express or implied.
 *
 ****************************************************************************/

#ifndef CODECCTX_H
#define CODECCTX_H

/**
 * external api header. 01/04/06 Marsdaddy <pajojo@gmail.com>
 */


#ifdef __cplusplus
extern "C" {
#endif

struct AVCodec;

enum CodecID {
    CODEC_ID_NONE,
    CODEC_ID_MPEG1VIDEO,
    CODEC_ID_MPEG2VIDEO, /* prefered ID for MPEG Video 1 or 2 decoding */
    CODEC_ID_MPEG2VIDEO_XVMC,
    CODEC_ID_H263,
    CODEC_ID_RV10,
    CODEC_ID_RV20,
    CODEC_ID_MP2,
    CODEC_ID_MP3, /* prefered ID for MPEG Audio layer 1, 2 or3 decoding */
    CODEC_ID_VORBIS,
    CODEC_ID_AC3,
    CODEC_ID_MJPEG,
    CODEC_ID_MJPEGB,
    CODEC_ID_LJPEG,
    CODEC_ID_SP5X,
    CODEC_ID_MPEG4,
    CODEC_ID_RAWVIDEO,
    CODEC_ID_MSMPEG4V1,
    CODEC_ID_MSMPEG4V2,
    CODEC_ID_MSMPEG4V3,
    CODEC_ID_WMV1,
    CODEC_ID_WMV2,
    CODEC_ID_H263P,
    CODEC_ID_H263I,
    CODEC_ID_FLV1,
    CODEC_ID_SVQ1,
    CODEC_ID_SVQ3,
    CODEC_ID_DVVIDEO,
    CODEC_ID_DVAUDIO,
    CODEC_ID_WMAV1,
    CODEC_ID_WMAV2,
    CODEC_ID_MACE3,
    CODEC_ID_MACE6,
    CODEC_ID_HUFFYUV,
    CODEC_ID_CYUV,
    CODEC_ID_H264,
    CODEC_ID_INDEO3,
    CODEC_ID_VP3,
    CODEC_ID_THEORA,
    CODEC_ID_AAC,
    CODEC_ID_MPEG4AAC,
    CODEC_ID_ASV1,
    CODEC_ID_ASV2,
    CODEC_ID_FFV1,
    CODEC_ID_4XM,
    CODEC_ID_VCR1,
    CODEC_ID_CLJR,
    CODEC_ID_MDEC,
    CODEC_ID_ROQ,
    CODEC_ID_INTERPLAY_VIDEO,
    CODEC_ID_XAN_WC3,
    CODEC_ID_XAN_WC4,
    CODEC_ID_RPZA,
    CODEC_ID_CINEPAK,
    CODEC_ID_WS_VQA,
    CODEC_ID_MSRLE,
    CODEC_ID_MSVIDEO1,
    CODEC_ID_IDCIN,
    CODEC_ID_8BPS,
    CODEC_ID_SMC,
    CODEC_ID_FLIC,
    CODEC_ID_TRUEMOTION1,
    CODEC_ID_VMDVIDEO,
    CODEC_ID_VMDAUDIO,
    CODEC_ID_MSZH,
    CODEC_ID_ZLIB,
    CODEC_ID_QTRLE,

    /* various pcm "codecs" */
    CODEC_ID_PCM_S16LE,
    CODEC_ID_PCM_S16BE,
    CODEC_ID_PCM_U16LE,
    CODEC_ID_PCM_U16BE,
    CODEC_ID_PCM_S8,
    CODEC_ID_PCM_U8,
    CODEC_ID_PCM_MULAW,
    CODEC_ID_PCM_ALAW,

    /* various adpcm codecs */
    CODEC_ID_ADPCM_IMA_QT,
    CODEC_ID_ADPCM_IMA_WAV,
    CODEC_ID_ADPCM_IMA_DK3,
    CODEC_ID_ADPCM_IMA_DK4,
    CODEC_ID_ADPCM_IMA_WS,
    CODEC_ID_ADPCM_IMA_SMJPEG,
    CODEC_ID_ADPCM_MS,
    CODEC_ID_ADPCM_4XM,
    CODEC_ID_ADPCM_XA,
    CODEC_ID_ADPCM_ADX,
    CODEC_ID_ADPCM_EA,

	/* AMR */
    CODEC_ID_AMR_NB,
    CODEC_ID_AMR_WB,

    /* RealAudio codecs*/
    CODEC_ID_RA_144,
    CODEC_ID_RA_288,

    /* various DPCM codecs */
    CODEC_ID_ROQ_DPCM,
    CODEC_ID_INTERPLAY_DPCM,
    CODEC_ID_XAN_DPCM,

    CODEC_ID_MPEG2TS, /* _FAKE_ codec to indicate a raw MPEG2 transport
                         stream (only used by libavformat) */
};


enum CodecType {
    CODEC_TYPE_UNKNOWN = -1,
    CODEC_TYPE_VIDEO,
    CODEC_TYPE_AUDIO,
    CODEC_TYPE_DATA,
};



/**
 * main external api structure.
 */
typedef struct CodecContext {
    /**
     * the average bitrate.
     * - encoding: set by user. unused for constant quantizer encoding
     * - decoding: set by lavc. 0 or some bitrate if this info is available in the stream
     */
    int bit_rate;

    int bits_per_sample;

    /**
     * number of bits the bitstream is allowed to diverge from the reference.
     *           the reference can be CBR (for CBR pass1) or VBR (for pass2)
     * - encoding: set by user. unused for constant quantizer encoding
     * - decoding: unused
     */
    int bit_rate_tolerance;

    /**
     * CODEC_FLAG_*.
     * - encoding: set by user.
     * - decoding: set by user.
     */
    int flags;

    /**
     * some codecs needs additionnal format info. It is stored here
     * - encoding: set by user.
     * - decoding: set by lavc. (FIXME is this ok?)
     */
    int sub_id;

    /**
     * motion estimation algorithm used for video coding.
     * - encoding: MUST be set by user.
     * - decoding: unused
     */
    int me_method;

    /**
     * some codecs need / can use extra-data like huffman tables.
     * mjpeg: huffman tables
     * rv10: additional flags
     * mpeg4: global headers (they can be in the bitstream or here)
     * - encoding: set/allocated/freed by lavc.
     * - decoding: set/allocated/freed by user.
     */
    void *extradata;
    int extradata_size;

    /* video only */
    /**
     * frames per sec multiplied by frame_rate_base.
     * for variable fps this is the precission, so if the timestamps
     * can be specified in msec precssion then this is 1000*frame_rate_base
     * - encoding: MUST be set by user
     * - decoding: set by lavc. 0 or the frame_rate if available
     */
    int frame_rate;

    /**
     * width / height.
     * - encoding: MUST be set by user.
     * - decoding: set by user if known, codec should override / dynamically change if needed
     */
    int width, height;

#define FF_ASPECT_SQUARE 1
#define FF_ASPECT_4_3_625 2
#define FF_ASPECT_4_3_525 3
#define FF_ASPECT_16_9_625 4
#define FF_ASPECT_16_9_525 5
#define FF_ASPECT_EXTENDED 15

    /**
     * the number of pictures in a group of pitures, or 0 for intra_only.
     * - encoding: set by user.
     * - decoding: unused
     */
    int gop_size;

    /**
     * Frame rate emulation. If not zero lower layer (i.e. format handler)
     * has to read frames at native frame rate.
     * - encoding: set by user.
     * - decoding: unused.
     */
    int rate_emu;

    /* audio only */
    int sample_rate; ///< samples per sec
    int channels;
    int sample_fmt;  ///< sample format, currenly unused

    /* the following data should not be initialized */
    int frame_size;     ///< in samples, initialized when calling 'init'
    int frame_number;   ///< audio or video frame number
    int real_pict_num;  ///< returns the real picture number of previous encoded frame

    struct AVCodec *codec;

    void *priv_data; // PJJ

    /* statistics, used for 2-pass encoding */
    int mv_bits;
    int header_bits;
    int i_tex_bits;
    int p_tex_bits;
    int i_count;
    int p_count;
    int skip_count;
    int misc_bits;

    /**
     * number of bits used for the previously encoded frame.
     * - encoding: set by lavc
     * - decoding: unused
     */
    int frame_bits;

    /**
     * private data of the user, can be used to carry app specific stuff.
     * - encoding: set by user
     * - decoding: set by user
     */
    void *opaque;

    char codec_name[32];
    enum CodecType codec_type; /* see CODEC_TYPE_xxx */
    enum CodecID codec_id; /* see CODEC_ID_xxx */

    /**
     * fourcc (LSB first, so "ABCD" -> ('D'<<24) + ('C'<<16) + ('B'<<8) + 'A').
     * this is used to workaround some encoder bugs
     * - encoding: set by user, if not then the default based on codec_id will be used
     * - decoding: set by user, will be converted to upper case by lavc during init
     */
    unsigned int codec_tag;

    /**
     * workaround bugs in encoders which sometimes cannot be detected automatically.
     * - encoding: unused
     * - decoding: set by user
     */
    int workaround_bugs;


    int block_align; ///< used by some WAV based audio codecs

    int parse_only; /* - decoding only: if true, only parsing is done
                       (function avcodec_parse_frame()). The frame
                       data is returned. Only MPEG codecs support this now. */

    /**
     * 0-> h263 quant 1-> mpeg quant.
     * - encoding: set by user.
     * - decoding: unused
     */
    int mpeg_quant;

    /**
     * pass1 encoding statistics output buffer.
     * - encoding: set by lavc
     * - decoding: unused
     */
    char *stats_out;

    /**
     * pass2 encoding statistics input buffer.
     * concatenated stuff from stats_out of pass1 should be placed here
     * - encoding: allocated/set/freed by user
     * - decoding: unused
     */
    char *stats_in;

    int rc_override_count;

    /**
     * rate control equation.
     * - encoding: set by user
     * - decoding: unused
     */
    char *rc_eq;

    uint64_t error[4];

    /**
     * minimum MB quantizer.
     * - encoding: set by user.
     * - decoding: unused
     */
    int mb_qmin;

    /**
     * maximum MB quantizer.
     * - encoding: set by user.
     * - decoding: unused
     */
    int mb_qmax;

    /**
     * motion estimation compare function.
     * - encoding: set by user.
     * - decoding: unused
     */
    int me_cmp;
    /**
     * subpixel motion estimation compare function.
     * - encoding: set by user.
     * - decoding: unused
     */
    int me_sub_cmp;
    /**
     * macroblock compare function (not supported yet).
     * - encoding: set by user.
     * - decoding: unused
     */
    int mb_cmp;
    /**
     * interlaced dct compare function
     * - encoding: set by user.
     * - decoding: unused
     */
    int ildct_cmp;
#define FF_CMP_SAD  0
#define FF_CMP_SSE  1
#define FF_CMP_SATD 2
#define FF_CMP_DCT  3
#define FF_CMP_PSNR 4
#define FF_CMP_BIT  5
#define FF_CMP_RD   6
#define FF_CMP_ZERO 7
#define FF_CMP_VSAD 8
#define FF_CMP_VSSE 9
#define FF_CMP_CHROMA 256

    /**
     * ME diamond size & shape.
     * - encoding: set by user.
     * - decoding: unused
     */
    int dia_size;

    /**
     * amount of previous MV predictors (2a+1 x 2a+1 square).
     * - encoding: set by user.
     * - decoding: unused
     */
    int last_predictor_count;

    /**
     * pre pass for motion estimation.
     * - encoding: set by user.
     * - decoding: unused
     */
    int pre_me;

    /**
     * motion estimation pre pass compare function.
     * - encoding: set by user.
     * - decoding: unused
     */
    int me_pre_cmp;

    /**
     * ME pre pass diamond size & shape.
     * - encoding: set by user.
     * - decoding: unused
     */
    int pre_dia_size;

    /**
     * subpel ME quality.
     * - encoding: set by user.
     * - decoding: unused
     */
    int me_subpel_quality;

    /**
     * Maximum motion estimation search range in subpel units.
     * if 0 then no limit
     *
     * - encoding: set by user.
     * - decoding: unused.
     */
    int me_range;

    /**
     * frame_rate_base.
     * for variable fps this is 1
     * - encoding: set by user.
     * - decoding: set by lavc.
     * @todo move this after frame_rate
     */

    int frame_rate_base;
    /**
     * intra quantizer bias.
     * - encoding: set by user.
     * - decoding: unused
     */
    int intra_quant_bias;
#define FF_DEFAULT_QUANT_BIAS 999999

    /**
     * inter quantizer bias.
     * - encoding: set by user.
     * - decoding: unused
     */
    int inter_quant_bias;

    /**
     * color table ID.
     * - encoding: unused.
     * - decoding: which clrtable should be used for 8bit RGB images
     *             table have to be stored somewhere FIXME
     */
    int color_table_id;

    /**
     * internal_buffer count.
     * Dont touch, used by lavc default_get_buffer()
     */
    int internal_buffer_count;

    /**
     * internal_buffers.
     * Dont touch, used by lavc default_get_buffer()
     */
    void *internal_buffer;

    /**
     * global quality for codecs which cannot change it per frame.
     * this should be proportional to MPEG1/2/4 qscale.
     * - encoding: set by user.
     * - decoding: unused
     */
    int global_quality;

#define FF_CODER_TYPE_VLC   0
#define FF_CODER_TYPE_AC    1
    /**
     * coder type
     * - encoding: set by user.
     * - decoding: unused
     */
    int coder_type;

    /**
     * context model
     * - encoding: set by user.
     * - decoding: unused
     */
    int context_model;

    /**
     * slice flags
     * - encoding: unused
     * - decoding: set by user.
     */
    int slice_flags;
#define SLICE_FLAG_CODED_ORDER    0x0001 ///< draw_horiz_band() is called in coded order instead of display
#define SLICE_FLAG_ALLOW_FIELD    0x0002 ///< allow draw_horiz_band() with field slices (MPEG2 field pics)
#define SLICE_FLAG_ALLOW_PLANE    0x0004 ///< allow draw_horiz_band() with 1 component at a time (SVQ1)

    /**
     * XVideo Motion Acceleration
     * - encoding: forbidden
     * - decoding: set by decoder
     */
    int xvmc_acceleration;

    /**
     * macroblock decision mode
     * - encoding: set by user.
     * - decoding: unused
     */
    int mb_decision;
#define FF_MB_DECISION_SIMPLE 0        ///< uses mb_cmp
#define FF_MB_DECISION_BITS   1        ///< chooses the one which needs the fewest bits
#define FF_MB_DECISION_RD     2        ///< rate distoration

    /**
     * custom intra quantization matrix
     * - encoding: set by user, can be NULL
     * - decoding: set by lavc
     */
    uint16_t *intra_matrix;

    /**
     * custom inter quantization matrix
     * - encoding: set by user, can be NULL
     * - decoding: set by lavc
     */
    uint16_t *inter_matrix;

    /**
     * fourcc from the AVI stream header (LSB first, so "ABCD" -> ('D'<<24) + ('C'<<16) + ('B'<<8) + 'A').
     * this is used to workaround some encoder bugs
     * - encoding: unused
     * - decoding: set by user, will be converted to upper case by lavc during init
     */
    unsigned int stream_codec_tag;

    /**
     * scene change detection threshold.
     * 0 is default, larger means fewer detected scene changes
     * - encoding: set by user.
     * - decoding: unused
     */
    int scenechange_threshold;

    /**
     * minimum lagrange multipler
     * - encoding: set by user.
     * - decoding: unused
     */
    int lmin;

    /**
     * maximum lagrange multipler
     * - encoding: set by user.
     * - decoding: unused
     */
    int lmax;

    /**
     * noise reduction strength
     * - encoding: set by user.
     * - decoding: unused
     */
    int noise_reduction;

    /**
     * simulates errors in the bitstream to test error concealment.
     * - encoding: set by user.
     * - decoding: unused.
     */
    int error_rate;

    /**
     * MP3 antialias algorithm, see FF_AA_* below.
     * - encoding: unused
     * - decoding: set by user
     */
    int antialias_algo;
#define FF_AA_AUTO    0
#define FF_AA_FASTINT 1 //not implemented yet
#define FF_AA_INT     2
#define FF_AA_FLOAT   3
    /**
     * Quantizer noise shaping.
     * - encoding: set by user
     * - decoding: unused
     */
    int quantizer_noise_shaping;
} CodecContext;


/**
 * AVOption.
 */
typedef struct AVOption {
    /** options' name */
    const char *name; /* if name is NULL, it indicates a link to next */
    /** short English text help or const struct AVOption* subpointer */
    const char *help; //	const struct AVOption* sub;
    /** offset to context structure where the parsed value should be stored */
    int offset;
    /** options' type */
    int type;
#define FF_OPT_TYPE_BOOL 1      ///< boolean - true,1,on  (or simply presence)
#define FF_OPT_TYPE_DOUBLE 2    ///< double
#define FF_OPT_TYPE_INT 3       ///< integer
#define FF_OPT_TYPE_STRING 4    ///< string (finished with \0)
#define FF_OPT_TYPE_MASK 0x1f	///< mask for types - upper bits are various flags
//#define FF_OPT_TYPE_EXPERT 0x20 // flag for expert option
#define FF_OPT_TYPE_FLAG (FF_OPT_TYPE_BOOL | 0x40)
#define FF_OPT_TYPE_RCOVERRIDE (FF_OPT_TYPE_STRING | 0x80)
    /** min value  (min == max   ->  no limits) */
    double min;
    /** maximum value for double/int */
    double max;
    /** default boo [0,1]l/double/int value */
    double defval;
    /**
     * default string value (with optional semicolon delimited extra option-list
     * i.e.   option1;option2;option3
     * defval might select other then first argument as default
     */
    const char *defstr;
#define FF_OPT_MAX_DEPTH 10
} AVOption;

/**
 * AVCodec.
 */
typedef struct AVCodec {
    const char *name;
    enum CodecType type;
    int id;
    int priv_data_size;
    int (*init)(CodecContext *);
    int (*encode)(CodecContext *, uint8_t *buf, int buf_size, void *data);
    int (*close)(CodecContext *);
    int (*decode)(CodecContext *, void *outdata, int *outdata_size,
                  uint8_t *buf, int buf_size);
    int capabilities;
    const AVOption *options;
    struct AVCodec *next;
    void (*flush)(CodecContext *);
} AVCodec;

#endif /* CODECCTX_H */
