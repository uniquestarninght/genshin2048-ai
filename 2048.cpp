#include <ctype.h>
#include <math.h>
#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <vector>
#include <algorithm>
#include "2048.h"
#include <unordered_map>
#include "config.h"
#if defined(HAVE_UNORDERED_MAP)
#define _CRT_SECURE_NO_WARNINGS
#include <unordered_map>
typedef std::unordered_map<board_t, trans_table_entry_t> trans_table_t;
#elif defined(HAVE_TR1_UNORDERED_MAP)
#include <tr1/unordered_map>
typedef std::tr1::unordered_map<board_t, trans_table_entry_t> trans_table_t;
#else
//#include <map>
//typedef std::map<board_t, trans_table_entry_t> trans_table_t;
#endif

/* MSVC compatibility: undefine max and min macros */
#if defined(max)
#undef max
#endif

#if defined(min)
#undef min
#endif
struct trans_table_key_t {
    board_t board;
    int prev_move;

    bool operator==(const trans_table_key_t& other) const {
        return board == other.board && prev_move == other.prev_move;
    }

};

struct trans_table_hash {
    std::hash<board_t> board_hasher;
    std::hash<int>     int_hasher;

    size_t operator()(const trans_table_key_t& k) const {
        return board_hasher(k.board) ^ (int_hasher(k.prev_move) << 1);
    }

};

using trans_table_t = std::unordered_map<board_t, trans_table_entry_t>;

// Transpose rows/columns in a board:
//   0123       048c
//   4567  -->  159d
//   89ab       26ae
//   cdef       37bf
static inline board_t transpose(board_t x)
{
    board_t a1 = x & 0xF0F00F0FF0F00F0FULL;
    board_t a2 = x & 0x0000F0F00000F0F0ULL;
    board_t a3 = x & 0x0F0F00000F0F0000ULL;
    board_t a = a1 | (a2 << 12) | (a3 >> 12);
    board_t b1 = a & 0xFF00FF0000FF00FFULL;
    board_t b2 = a & 0x00FF00FF00000000ULL;
    board_t b3 = a & 0x00000000FF00FF00ULL;
    return b1 | (b2 >> 24) | (b3 << 24);
}

// Count the number of empty positions (= zero nibbles) in a board.
// Precondition: the board cannot be fully empty.
static int count_empty(board_t x)
{
    x |= (x >> 2) & 0x3333333333333333ULL;
    x |= (x >> 1);
    x = ~x & 0x1111111111111111ULL;
    // At this point each nibble is:
    //  0 if the original nibble was non-zero
    //  1 if the original nibble was zero
    // Next sum them all
    x += x >> 32;
    x += x >> 16;
    x += x >> 8;
    x += x >> 4; // this can overflow to the next nibble if there were 16 empty positions
    return x & 0xf;
}
static bool has_2048_moved(board_t old_board, board_t new_board) {
    // 比较 2048 (rank=11) 的位置是否变化
    board_t mask_2048_old = 0;
    board_t mask_2048_new = 0;
    for (int i = 0; i < 16; ++i) {
        if (((old_board >> (4 * i)) & 0xF) == 11) {
            mask_2048_old |= (1ULL << (4 * i));
        }
        if (((new_board >> (4 * i)) & 0xF) == 11) {
            mask_2048_new |= (1ULL << (4 * i));
        }
    }
    return mask_2048_old != mask_2048_new;
}
/* We can perform state lookups one row at a time by using arrays with 65536 entries. */

/* Move tables. Each row or compressed column is mapped to (oldrow^newrow) assuming row/col 0.
 *

 * Thus, the value is 0 if there is no move, and otherwise equals a value that can easily be
 * xor'ed into the current board state to update the board. */
static row_t row_left_table[65536];
static row_t row_right_table[65536];
static board_t col_up_table[65536];
static board_t col_down_table[65536];
static float heur_score_table[65536];
static float score_table[65536];

// Heuristic scoring settings
static const float SCORE_LOST_PENALTY = 200000.0f;
static const float SCORE_MONOTONICITY_POWER = 4.0f;
static const float SCORE_MONOTONICITY_WEIGHT = 47.0f;
static const float SCORE_SUM_POWER = 3.5f;
static const float SCORE_SUM_WEIGHT = 11.0f;
static const float SCORE_MERGES_WEIGHT = 700.0f;
static const float SCORE_EMPTY_WEIGHT = 270.0f;

void init_tables() {
    for (unsigned row = 0; row < 65536; ++row) {
        unsigned line[4] = {
                (row >> 0) & 0xf,
                (row >> 4) & 0xf,
                (row >> 8) & 0xf,
                (row >> 12) & 0xf
        };

        // Score
        float score = 0.0f;
        for (int i = 0; i < 4; ++i) {
            int rank = line[i];
            if (rank >= 2)
            {
                // the score is the total sum of the tile and all intermediate merged tiles
                score += (rank - 1) * (1 << rank);
            }


        }
        score_table[row] = score;


        // Heuristic score
        float sum = 0;
        int empty = 0;
        int merges = 0;

        int prev = 0;
        int counter = 0;
        for (int i = 0; i < 4; ++i) {
            int rank = line[i];
            sum += pow(rank, SCORE_SUM_POWER);
            if (rank == 0) {
                empty++;
            }
            else {
                if (prev == rank) {
                    counter++;
                }
                else if (counter > 0) {
                    merges += 1 + counter;
                    counter = 0;
                }
                prev = rank;
            }
        }
        if (counter > 0) {
            merges += 1 + counter;
        }

        float monotonicity_left = 0;
        float monotonicity_right = 0;
        for (int i = 1; i < 4; ++i) {
            if (line[i - 1] > line[i]) {
                monotonicity_left += pow(line[i - 1], SCORE_MONOTONICITY_POWER) - pow(line[i], SCORE_MONOTONICITY_POWER);
            }
            else {
                monotonicity_right += pow(line[i], SCORE_MONOTONICITY_POWER) - pow(line[i - 1], SCORE_MONOTONICITY_POWER);
            }
        }

        heur_score_table[row] = SCORE_LOST_PENALTY +
            SCORE_EMPTY_WEIGHT * empty +
            SCORE_MERGES_WEIGHT * merges -
            SCORE_MONOTONICITY_WEIGHT * std::min(monotonicity_left, monotonicity_right) -
            SCORE_SUM_WEIGHT * sum;

        // execute a move to the left
        for (int i = 0; i < 3; ++i) {
            int j;
            for (j = i + 1; j < 4; ++j) {
                if (line[j] != 0) break;
            }
            if (j == 4) break; // no more tiles to the right

            if (line[i] == 0) {
                line[i] = line[j];
                line[j] = 0;
                i--; // retry this entry
            }
            else if (line[i] == line[j]) {
                //if(line[i] <= 0xf) {//原代码逻辑是不能大于2^15,也就是理论最大数字,但实际上,本人认为是2^16最大,这里注释掉是因为原神最大2048
                if (line[i] < 11)//原神限制了2048不让合成
                {
                    /* Pretend that 32768 + 32768 = 32768 (representational limit). */
                    line[i]++;
                    line[j] = 0;
                }
                //line[j] = 0;//移动到if里面.大于等于2048不会合并
            }
        }

        row_t result = (line[0] << 0) |
            (line[1] << 4) |
            (line[2] << 8) |
            (line[3] << 12);
        row_t rev_result = reverse_row(result);
        unsigned rev_row = reverse_row(row);

        row_left_table[row] = row ^ result;
        row_right_table[rev_row] = rev_row ^ rev_result;
        col_up_table[row] = unpack_col(row) ^ unpack_col(result);
        col_down_table[rev_row] = unpack_col(rev_row) ^ unpack_col(rev_result);
    }

}

static inline board_t execute_move_0(board_t board) {
    board_t ret = board;
    board_t t = transpose(board);
    ret ^= col_up_table[(t >> 0) & ROW_MASK] << 0;
    ret ^= col_up_table[(t >> 16) & ROW_MASK] << 4;
    ret ^= col_up_table[(t >> 32) & ROW_MASK] << 8;
    ret ^= col_up_table[(t >> 48) & ROW_MASK] << 12;
    return ret;
}

static inline board_t execute_move_1(board_t board) {
    board_t ret = board;
    board_t t = transpose(board);
    ret ^= col_down_table[(t >> 0) & ROW_MASK] << 0;
    ret ^= col_down_table[(t >> 16) & ROW_MASK] << 4;
    ret ^= col_down_table[(t >> 32) & ROW_MASK] << 8;
    ret ^= col_down_table[(t >> 48) & ROW_MASK] << 12;
    return ret;
}

static inline board_t execute_move_2(board_t board) {
    board_t ret = board;
    ret ^= board_t(row_left_table[(board >> 0) & ROW_MASK]) << 0;
    ret ^= board_t(row_left_table[(board >> 16) & ROW_MASK]) << 16;
    ret ^= board_t(row_left_table[(board >> 32) & ROW_MASK]) << 32;
    ret ^= board_t(row_left_table[(board >> 48) & ROW_MASK]) << 48;
    return ret;
}

static inline board_t execute_move_3(board_t board) {
    board_t ret = board;
    ret ^= board_t(row_right_table[(board >> 0) & ROW_MASK]) << 0;
    ret ^= board_t(row_right_table[(board >> 16) & ROW_MASK]) << 16;
    ret ^= board_t(row_right_table[(board >> 32) & ROW_MASK]) << 32;
    ret ^= board_t(row_right_table[(board >> 48) & ROW_MASK]) << 48;
    return ret;
}

/* Execute a move. */
board_t execute_move(int move, board_t board) {
    switch (move) {
    case 0: // up
        return execute_move_0(board);
    case 1: // down
        return execute_move_1(board);
    case 2: // left
        return execute_move_2(board);
    case 3: // right
        return execute_move_3(board);
    default:
        return ~0ULL;
    }
}

static inline int get_max_rank(board_t board) {
    int maxrank = 0;
    while (board) {
        maxrank = std::max(maxrank, int(board & 0xf));
        board >>= 4;
    }
    return maxrank;
}

static inline int count_distinct_tiles(board_t board) {
    uint16_t bitset = 0;     // 记录所有非空、非 2048 的 rank
    int count_2048 = 0;      // 单独统计 2048 数量
    int sum = 0;
    bool is_exist = false;
    bool has_512 = false;
    bool has_256 = false;
    bool has_128 = false;
    while (board) {
        int rank = board & 0xf;
        if (rank == 0) {
            // skip empty
        }
        else if (rank == 11) {  // 2048
            count_2048++;
        }
        else {
            bitset |= 1 << rank;
            if (rank == 10) { is_exist = true; }
            else if (rank == 9)has_512 = true;
            else if (rank == 8)has_256 = true;

        }
        board >>= 4;
    }
    // 统计非 2048 的种类数（去掉空格影响）
    bitset >>= 1;  // 移除 rank=0 的位
    int distinct_others = 0;
    uint16_t tmp = bitset;
    while (tmp) {
        tmp &= tmp - 1;
        distinct_others++;
    }
    sum = distinct_others + count_2048 - 1;
    if (count_2048 > 5) {
        sum = sum + distinct_others * 4 + 3;
        if (sum >= 39) {
            sum = 39;
        }
        if (is_exist) {
            sum += 3;
            if (has_512) {
                sum += 5;
                if (has_256) {
                    sum += 14;
                    if (has_128)sum += 10;
                }

            }
        }
        if (sum <= 32)sum = 32;
    }
    else if (count_2048 > 3) {
        sum = sum + distinct_others / 2 + 1;
        if (is_exist)sum = sum + count_2048 * 2 - 5;
        if ((count_2048 > 4)) {
            sum = 2 + sum + distinct_others * 3 / 2 + 1;
            if (has_512) {
                sum += 2;
                if (is_exist)sum = 28;
            }
        }
        if (sum >= 28) { sum = 28; }
    }
    return sum;

}


/* Optimizing the game */

struct eval_state {
    trans_table_t trans_table; // ← 现在这个类型已在上面正确定义
    int maxdepth;
    int curdepth;
    int cachehits;
    unsigned long moves_evaled;
    int depth_limit;

    eval_state() : maxdepth(0), curdepth(0), cachehits(0), moves_evaled(0), depth_limit(0) {}

};

// score a single board heuristically
static float score_heur_board(board_t board);
// score a single board actually (adding in the score from spawned 4 tiles)
static float score_board(board_t board);
// score over all possible moves
static float score_move_node(eval_state& state, board_t board, float cprob);
// score over all possible tile choices and placements
static float score_tilechoose_node(eval_state& state, board_t board, float cprob, int move);


static float score_helper(board_t board, const float* table) {
    return table[(board >> 0) & ROW_MASK] +
        table[(board >> 16) & ROW_MASK] +
        table[(board >> 32) & ROW_MASK] +
        table[(board >> 48) & ROW_MASK];
}
static int i = 0;
static bool is_perfect_endgame(board_t board) {
    int count_2048 = 0;
    /*   if (tag)
       {
           printf("is_perfect_endgame被score_heur_board调用%d次\n", i++);
       }
       else
           printf("被其他东西调用");*/
    uint16_t present_ranks = 0; // bit i 表示 rank i 是否存在（i >= 1）

    for (int i = 0; i < 16; ++i) {
        int rank = (board >> (4 * i)) & 0xF;
        if (rank == 0) continue;
        if (rank == 11) {
            count_2048++;
        }
        else if (rank >= 1 && rank <= 10) {
            present_ranks |= (1U << rank);
        }
        // 如果出现 rank > 11（比如 4096），则不符合“只有2048及以下”的要求
        else if (rank > 11) {
            return false;
        }
    }
    //printf("2048个数%d\n",count_2048);
    // 必须恰好有 6 个 2048
    if (count_2048 != 6) return false;

    // 检查 rank 1 到 10 是否全部存在（即 2, 4, ..., 1024）
    uint16_t required = (1U << 11) - 2; // bits 1~10 set: 0b11111111110 = 2046
    //int res =( (present_ranks & required) == required);
    //printf("required:%d,result:%d.\n",required,res);
    //return res;
    return (present_ranks & required) == required;

}

static float score_heur_board(board_t board) {
    // 检查是否达成“完美终局”
    //int num[4][4];
    //int pos = 0;
    //for (int i = 0; i < 4; i++) {
    //    for (int j = 0; j < 4; j++) {
    //        // 每次提取4位
    //        num[i][j] = (board >> (4 * pos)) & 0xF;
    //        pos++;
    //        printf("%d\t", num[i][j]);
    //    }
        //printf("第一行\n");
    //}
    //int tag = 1;
    if (is_perfect_endgame(board)) {
        //printf("perfect_endgame!!!!\n");
        return 100000000.0f;
    }
    //printf("if语句执行异常%d\n",is_perfect_endgame(board));


    return score_helper(board, heur_score_table) +
        score_helper(transpose(board), heur_score_table);

}

static float score_board(board_t board) {
    return score_helper(board, score_table);
}

// Statistics and controls
// cprob: cumulative probability
// don't recurse into a node with a cprob less than this threshold
static const float CPROB_THRESH_BASE = 0.001f;
static const int CACHE_DEPTH_LIMIT = 100;

static float score_tilechoose_node(eval_state& state, board_t board, float cprob, int prev_move = 4) {
    if (cprob < CPROB_THRESH_BASE || state.curdepth >= state.depth_limit) {
        return score_heur_board(board);
    }

    // === 仍然根据 prev_move 决定生成位置（原神规则）===
    std::vector<int> valid_positions;
    if (prev_move == 4) {
        for (int i = 0; i < 16; ++i)
            if (((board >> (4 * i)) & 0xF) == 0) valid_positions.push_back(i);
    }
    else {
        switch (prev_move) {
        case 0: for (int col = 0;col < 4;++col) if (!((board >> (4 * (12 + col))) & 0xF)) valid_positions.push_back(12 + col); break;
        case 1: for (int col = 0;col < 4;++col) if (!((board >> (4 * col)) & 0xF)) valid_positions.push_back(col); break;
        case 2: for (int row = 0;row < 4;++row) if (!((board >> (4 * (row * 4 + 3))) & 0xF)) valid_positions.push_back(row * 4 + 3); break;
        case 3: for (int row = 0;row < 4;++row) if (!((board >> (4 * (row * 4))) & 0xF)) valid_positions.push_back(row * 4); break;
        }
    }

    if (valid_positions.empty()) return score_heur_board(board);

    float total = 0.0f;
    float w = 1.0f / valid_positions.size();
    for (int pos : valid_positions) {
        board_t newboard = board | (1ULL << (4 * pos));
        // ↓ 关键：调用 score_move_node（这里会缓存！）
        total += w * score_move_node(state, newboard, cprob);
    }
    return total;

}

static bool should_punish_move_2048(board_t old_board, board_t new_board) {
    int count_old = 0, count_new = 0;
    board_t mask_old = 0, mask_new = 0;

    for (int i = 0; i < 16; ++i) {
        if (((old_board >> (4 * i)) & 0xF) == 11) {
            count_old++;
            mask_old |= (1ULL << (4 * i));
        }
        if (((new_board >> (4 * i)) & 0xF) == 11) {
            count_new++;
            mask_new |= (1ULL << (4 * i));
        }
    }

    // 情况1: 合成了新的2048（数量增加）→ 绝对不惩罚！
    if (count_new > count_old) {
        return false;
    }

    // 情况2: 2048数量不变，但位置变了 → 移动了已有2048 → 惩罚！
    if (count_old > 0 && mask_old != mask_new) {
        return true;
    }

    // 其他情况（无2048，或位置没变）→ 不惩罚
    return false;

}
static float score_move_node(eval_state& state, board_t board, float cprob) {
    // === 每次进入 Max 节点都更新 maxdepth ===
    state.maxdepth = std::max(state.maxdepth, state.curdepth);

    if (state.curdepth < CACHE_DEPTH_LIMIT) {
        auto it = state.trans_table.find(board);
        if (it != state.trans_table.end() && it->second.depth <= state.curdepth) {
            state.cachehits++;
            return it->second.heuristic;
        }
    }

    float best = 0.0f;
    state.curdepth++;

    bool found_valid = false; // 标记是否有合法移动
    for (int move = 0; move < 4; ++move) {
        board_t newboard = execute_move(move, board);
        state.moves_evaled++;
        if (newboard == board)
        {
            bool perfect = is_perfect_endgame(newboard);
            if (perfect == false) {
                continue;
            }
        }
        else if (should_punish_move_2048(board, newboard)) continue;

        found_valid = true;
        best = std::max(best, score_tilechoose_node(state, newboard, cprob, move));
    }

    // 如果没有合法移动，也要更新 maxdepth（当前深度就是叶子）
    if (!found_valid) {
        state.maxdepth = std::max(state.maxdepth, state.curdepth);
    }

    state.curdepth--;

    if (state.curdepth < CACHE_DEPTH_LIMIT) {
        state.trans_table[board] = { static_cast<uint8_t>(state.curdepth), best };
    }

    return best;

}

static float _score_toplevel_move(eval_state& state, board_t board, int move) {
    board_t newboard = execute_move(move, board);
    if (board == newboard) return 0;
    if (should_punish_move_2048(board, newboard)) return -1e6f;
    // 调用 chance 节点，传入 move 作为 prev_move
    return score_tilechoose_node(state, newboard, 1.0f, move) + 1e-6f;
}

float score_toplevel_move(board_t board, int move) {
    float res;
    struct timeval start, finish;
    double elapsed;
    eval_state state;
    state.depth_limit = std::max(3, count_distinct_tiles(board) - 2);

    gettimeofday(&start, NULL);
    res = _score_toplevel_move(state, board, move);
    gettimeofday(&finish, NULL);

    elapsed = (finish.tv_sec - start.tv_sec);
    elapsed += (finish.tv_usec - start.tv_usec) / 1000000.0;

    printf("Move %d: result %f: eval'd %ld moves (%d cache hits, %d cache size) in %.2f seconds (maxdepth=%d)\n", move, res,
        state.moves_evaled, state.cachehits, (int)state.trans_table.size(), elapsed, state.maxdepth);

    return res;

}

/* Find the best move for a given board. */
int find_best_move(board_t board) {
    int legal_moves[4];
    int num_legal = 0;

    // 先找出所有合法移动
    for (int move = 0; move < 4; ++move) {
        board_t new_board = execute_move(move, board);
        if (new_board != board) {
            legal_moves[num_legal++] = move;
        }
        else {
            if (should_punish_move_2048(board, new_board) == false)legal_moves[num_legal++] = move;
        }
    }
    // 如果没有合法移动，返回 -1（游戏结束）
    if (num_legal == 0) {
        print_board(board);
        printf("No legal moves. Game over.\n");
        return -1;
    }
    // 如果只有一个合法移动，直接返回，不搜索
    if (num_legal == 1) {
        print_board(board);
        printf("Only one legal move: %c\n", "UDLR"[legal_moves[0]]);
        return legal_moves[0];
    }
    // 否则，正常进行搜索
    float best = 0;
    int bestmove = -1;

    print_board(board);
    printf("Current scores: heur %.0f, actual %.0f\n", score_heur_board(board), score_board(board));

    for (int i = 0; i < num_legal; ++i) {
        int move = legal_moves[i];
        float res = score_toplevel_move(board, move);

        if (res > best) {
            best = res;
            bestmove = move;
        }
    }

    return bestmove;

}

int ask_for_move(board_t board) {
    int move;
    char validstr[5];
    char* validpos = validstr;

    print_board(board);

    for (move = 0; move < 4; move++) {
        if (execute_move(move, board) != board)
            *validpos++ = "UDLR"[move];
    }
    *validpos = 0;
    if (validpos == validstr)
        return -1;

    while (1) {
        char movestr[64];
        const char* allmoves = "UDLR";

        printf("Move [%s]? ", validstr);

        if (!fgets(movestr, sizeof(movestr) - 1, stdin))
            return -1;

        if (!strchr(validstr, toupper(movestr[0]))) {
            printf("Invalid move.\n");
            continue;
        }

        return strchr(allmoves, toupper(movestr[0])) - allmoves;
    }

}

/* Playing the game */
static board_t draw_tile() {
    return 1;//原代码是return (unif_random(10) < 9) ? 1 : 2,10%的概率是4,但是原神的游戏只会生成2,所以算法可以改成只返回2
}

static board_t insert_tile_rand(board_t board, board_t tile, int move = 4) {
    int count = count_empty(board);
    if (count == 0)return board;
    if (move == 4)
    {
        int index = unif_random(count);
        board_t tmp = board;
        while (true) {
            while ((tmp & 0xf) != 0) {
                tmp >>= 4;
                tile <<= 4;
            }
            if (index == 0) break;
            --index;
            tmp >>= 4;
            tile <<= 4;
        }
        return board | tile;
    }
    else
    {
        int indexs[4] = { 0,0,0,0 };
        int i = 0;
        int index;
        switch (move) {
        case 0: // up - 生成在最下面的空位            
            for (int col = 0; col < 4; col++) {
                int pos = 12 + col;
                if (((board >> (4 * pos)) & 0xF) == 0) {
                    indexs[i++] = 4 * pos;
                }
            }
            index = unif_random(i);
            return board | (tile << indexs[index]);
            break;
        case 1: // down - 生成在最上面的空位           
            for (int col = 0; col < 4; col++) {
                if (((board >> (4 * col)) & 0xF) == 0) {
                    indexs[i++] = 4 * col;
                }
            }
            index = unif_random(i);
            return board | (tile << indexs[index]);
            break;
        case 2: // left - 生成在最右侧的空位           
            for (int row = 0; row < 4; row++) {
                int pos = row * 4 + 3;
                if (((board >> (4 * pos)) & 0xF) == 0) {
                    indexs[i++] = 4 * pos;
                }
            }
            index = unif_random(i);
            return board | (tile << indexs[index]);
            break;
        case 3: // right - 生成在最左侧的空位           
            for (int row = 0; row < 4; row++) {
                int pos = row * 4;
                if (((board >> (4 * pos)) & 0xF) == 0) {
                    indexs[i++] = 4 * pos;
                }
            }
            index = unif_random(i);
            return board | (tile << indexs[index]);
            break;
        default:
            return board;
        }
    }
}

static board_t initial_board() {
    board_t board = draw_tile() << (4 * unif_random(16));
    return insert_tile_rand(board, draw_tile());
}

void play_game(get_move_func_t get_move) {
    board_t board = initial_board();
    int moveno = 0;
    int scorepenalty = 0; // "penalty" for obtaining free 4 tiles
    while (1) {
        int move;
        board_t newboard;

        for (move = 0; move < 4; move++) {
            if (execute_move(move, board) != board)
                break;
        }
        if (move == 4)
            break; // no legal moves

        printf("\nMove #%d, current score=%.0f\n", ++moveno, score_board(board) - scorepenalty);

        move = get_move(board);
        if (move < 0)
            break;

        newboard = execute_move(move, board);
        if (newboard == board) {
            printf("Illegal move!\n");
            moveno--;
            continue;
        }

        board_t tile = draw_tile();
        //if (tile == 2) scorepenalty += 4;
        board = insert_tile_rand(newboard, tile, move);
        //system("pause");
    }

    print_board(board);
    printf("\nGame over. Your score is %.0f. The highest rank you achieved was %d.\n", score_board(board) - scorepenalty, get_max_rank(board));
    system("pause");

}
static int tile_to_rank(uint64_t tile) {
    if (tile == 0) return 0;
    int rank = 0;
    while (tile > 1) {
        if (tile % 2 != 0) {
            // 非 2 的幂，视为无效
            printf("Warning: %llu is not a power of two!\n", (unsigned long long)tile);
            return 0;
        }
        tile /= 2;
        rank++;
    }
    // 限制最大 rank 为 15（防止溢出）
    return (rank <= 15) ? rank : 15;
}

// 调试函数：让用户输入 16 个 tile 值，AI 给出一步建议
void debug_solve_one_step() {
    printf("\n=== 2048 AI 单步求解器 ===\n");
    printf("请输入 16 个空格分隔的数字（0 表示空格）\n");
    printf("> ");

    char line[1024];
    if (!fgets(line, sizeof(line), stdin)) {
        printf("输入失败。\n");
        return;
    }

    uint64_t tiles[16];
    int count = 0;
    char* p = line;

    while (count < 16 && *p) {
        // 跳过空白字符
        while (*p == ' ' || *p == '\t' || *p == '\n' || *p == '\r') {
            p++;
        }
        if (*p == '\0') break;

        char* endptr;
        unsigned long long val = strtoull(p, &endptr, 10);
        if (endptr == p) { // 没有解析到数字
            printf("无效数字在位置: '%s'\n", p);
            return;
        }
        tiles[count++] = (uint64_t)val;
        p = endptr;
    }

    if (count != 16) {
        printf("错误：请恰好输入 16 个数字！当前输入了 %d 个。\n", count);
        return;
    }

    // 构造 board_t
    board_t board = 0;
    for (int i = 0; i < 16; i++) {
        int rank = tile_to_rank(tiles[i]);
        board |= ((board_t)rank) << (4 * i);
    }

    printf("\n解析后的棋盘:\n");
    print_board(board);

    /*  if (is_perfect_endgame(board)) {
          printf(" 完美终局达成！\n");
          return;
      }*/

    int move = find_best_move(board);
    if (move == -2) {
        printf(" AI 检测到完美终局！\n");
    }
    else if (move == -1) {
        printf("无合法移动。\n");
    }
    else if (move >= 0 && move <= 3) {
        const char* dirs = "UDLR";
        const char* names[] = {"","","",""};
        printf("AI 建议: %c (%s)\n", dirs[move], names[move]);
    }
    else {
        printf("未知返回值: %d\n", move);
    }

}
//int main() {
    //init_tables();
    //play_game(find_best_move);
//}
int main() {
    init_tables();

    //printf("2048 AI 求解器\n");
    //printf("1. 正常游戏模式\n");
    //printf("2. 调试模式：输入自定义棋盘，AI 给出单步建议\n");
    //printf("请选择 (1 或 2): ");

    //char choice;
    //scanf_s("%c", &choice);
    //getchar(); // 吃掉回车符，避免影响后续 fgets

    //if (choice == '2') {
        //debug_solve_one_step(); // 调用你刚写的调试函数
    //}
    //else {
    play_game(find_best_move);
    //}

    return 0;

}