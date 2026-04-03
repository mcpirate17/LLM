#include "../include/graph_analysis.h"

#include <ctype.h>
#include <math.h>
#include <stdint.h>
#include <stdlib.h>

typedef struct {
  const char* current;
} aria_formula_parser_t;

static void aria_skip_ws(aria_formula_parser_t* parser) {
  while (parser != NULL && parser->current != NULL &&
         isspace((unsigned char)*parser->current)) {
    parser->current += 1;
  }
}

static int aria_match_char(aria_formula_parser_t* parser, char expected) {
  aria_skip_ws(parser);
  if (parser->current == NULL || *parser->current != expected) {
    return 0;
  }
  parser->current += 1;
  return 1;
}

static int aria_parse_expression(aria_formula_parser_t* parser, double* out_value);

static int aria_parse_number(aria_formula_parser_t* parser, double* out_value) {
  char* end_ptr = NULL;

  aria_skip_ws(parser);
  if (parser == NULL || parser->current == NULL || out_value == NULL) {
    return -1;
  }

  *out_value = strtod(parser->current, &end_ptr);
  if (end_ptr == parser->current) {
    return -1;
  }

  parser->current = end_ptr;
  return 0;
}

static int aria_parse_primary(aria_formula_parser_t* parser, double* out_value) {
  if (aria_match_char(parser, '(')) {
    if (aria_parse_expression(parser, out_value) != 0) {
      return -1;
    }
    return aria_match_char(parser, ')') ? 0 : -1;
  }
  return aria_parse_number(parser, out_value);
}

static int aria_parse_unary(aria_formula_parser_t* parser, double* out_value) {
  if (aria_match_char(parser, '-')) {
    if (aria_parse_unary(parser, out_value) != 0) {
      return -1;
    }
    *out_value = -*out_value;
    return 0;
  }
  return aria_parse_primary(parser, out_value);
}

static int aria_parse_power(aria_formula_parser_t* parser, double* out_value) {
  double base = 0.0;
  double exponent = 0.0;
  const char* saved = NULL;

  if (aria_parse_unary(parser, &base) != 0) {
    return -1;
  }

  aria_skip_ws(parser);
  saved = parser->current;
  if (saved == NULL || saved[0] != '*' || saved[1] != '*') {
    *out_value = base;
    return 0;
  }

  parser->current += 2;
  if (aria_parse_power(parser, &exponent) != 0) {
    return -1;
  }

  *out_value = pow(base, exponent);
  return isfinite(*out_value) ? 0 : -1;
}

static int aria_parse_term(aria_formula_parser_t* parser, double* out_value) {
  double lhs = 0.0;

  if (aria_parse_power(parser, &lhs) != 0) {
    return -1;
  }

  while (1) {
    double rhs = 0.0;

    aria_skip_ws(parser);
    if (parser->current == NULL) {
      return -1;
    }

    if (parser->current[0] == '*' && parser->current[1] == '*') {
      break;
    }

    if (parser->current[0] == '*' && parser->current[1] != '\0') {
      parser->current += 1;
      if (aria_parse_power(parser, &rhs) != 0) {
        return -1;
      }
      lhs *= rhs;
      continue;
    }

    if (parser->current[0] == '/' && parser->current[1] == '/') {
      parser->current += 2;
      if (aria_parse_power(parser, &rhs) != 0 || rhs == 0.0) {
        return -1;
      }
      lhs = floor(lhs / rhs);
      continue;
    }

    if (parser->current[0] == '/') {
      parser->current += 1;
      if (aria_parse_power(parser, &rhs) != 0 || rhs == 0.0) {
        return -1;
      }
      lhs /= rhs;
      continue;
    }

    break;
  }

  *out_value = lhs;
  return isfinite(*out_value) ? 0 : -1;
}

static int aria_parse_expression(aria_formula_parser_t* parser, double* out_value) {
  double lhs = 0.0;

  if (aria_parse_term(parser, &lhs) != 0) {
    return -1;
  }

  while (1) {
    double rhs = 0.0;

    aria_skip_ws(parser);
    if (parser->current == NULL) {
      return -1;
    }

    if (parser->current[0] == '+') {
      parser->current += 1;
      if (aria_parse_term(parser, &rhs) != 0) {
        return -1;
      }
      lhs += rhs;
      continue;
    }

    if (parser->current[0] == '-') {
      parser->current += 1;
      if (aria_parse_term(parser, &rhs) != 0) {
        return -1;
      }
      lhs -= rhs;
      continue;
    }

    break;
  }

  *out_value = lhs;
  return isfinite(*out_value) ? 0 : -1;
}

int32_t aria_eval_param_formula(const char* formula, int64_t* out_value) {
  aria_formula_parser_t parser;
  double value = 0.0;

  if (formula == NULL || out_value == NULL) {
    return -1;
  }

  parser.current = formula;
  if (aria_parse_expression(&parser, &value) != 0) {
    return -1;
  }

  aria_skip_ws(&parser);
  if (parser.current == NULL || *parser.current != '\0' || !isfinite(value)) {
    return -1;
  }

  *out_value = (int64_t)value;
  return 0;
}
