"""LifelongAgentBench 官方 DirectTypeAnswerValidator（commit d6f19b42eb358d9150379f0c68c2985c5a867520，
src/tasks/instance/db_bench/task.py）原文，仅作为测试中的参照实现，用于核对 evaluate.py 的判分结果。"""
import math
import re
from typing import Optional, Sequence


class DBBenchType:
    RowValue = int | str | float
    Row = tuple[RowValue, ...]
    AnswerDirect = Sequence["DBBenchType.Row"]


class DirectTypeAnswerValidator:
    @staticmethod
    def _is_two_float_equal(float1: float, float2: float) -> bool:
        return math.isclose(float1, float2, rel_tol=1e-06, abs_tol=1e-06)

    @staticmethod
    def _get_tuple_position_list(answer: str) -> Optional[list[tuple[int, int]]]:
        # Return None if parsing fails
        tuple_position_list = []
        start_index: Optional[int] = None
        end_index: Optional[int] = None
        disable_flag = False
        disable_word = "Decimal"
        for index, char in enumerate(answer):
            if char == "(":
                if disable_word == answer[index - len(disable_word) : index]:
                    if disable_flag:
                        return None
                    disable_flag = True
                    continue
                else:
                    start_index = index
            elif char == ")":
                if disable_flag:
                    disable_flag = False
                    continue
                end_index = index
                if start_index is None:
                    return None
                tuple_position_list.append((start_index, end_index))
                start_index = None
                end_index = None
            else:
                pass
                # Can add further validation here
        if start_index is not None or end_index is not None:
            return None
        return tuple_position_list

    @staticmethod
    def _validate_single_tuple_str(
        agent_tuple_str: str, ground_truth_tuple: DBBenchType.Row
    ) -> bool:
        # region Get agent_answer_element_list
        raw_agent_answer_element_list = re.split(r"\s*,\s*", agent_tuple_str)
        agent_answer_element_list = []
        for element in raw_agent_answer_element_list:
            if not element.strip() == "":
                agent_answer_element_list.append(element.strip())
        # endregion
        # region Validate the length
        if len(agent_answer_element_list) != len(ground_truth_tuple):
            return False
        # endregion
        # region Validate a single element
        for agent_answer_element, ground_truth_element in zip(
            agent_answer_element_list, ground_truth_tuple
        ):
            if isinstance(ground_truth_element, (int, float)):
                # region Handle int and float
                agent_answer_element = re.sub(
                    r'^Decimal\((["\']?)(.*?)(["\']?)\)$', r"\2", agent_answer_element
                )
                # region Handle case """'-55'"""
                start_index = 0
                valid_char_in_float_str = "0123456789.-"
                for i in range(len(agent_answer_element)):
                    if agent_answer_element[i] in valid_char_in_float_str:
                        break
                    start_index = i
                end_index = len(agent_answer_element)
                for i in range(len(agent_answer_element) - 1, -1, -1):
                    if agent_answer_element[i] in valid_char_in_float_str:
                        break
                    end_index = i + 1
                # endregion
                try:
                    processed_element = float(
                        agent_answer_element[start_index:end_index]
                    )
                except:  # noqa
                    return False
                if not DirectTypeAnswerValidator._is_two_float_equal(
                    processed_element, ground_truth_element
                ):
                    return False
                # endregion
            elif isinstance(ground_truth_element, str):
                # region Handle str
                if 0 < len(agent_answer_element) - len(ground_truth_element) <= 2:
                    head_strip_flag = False
                    tail_strip_flag = False
                    for char in ["'", '"']:
                        if (
                            agent_answer_element.startswith(char)
                            and not ground_truth_element.startswith(char)
                            and not head_strip_flag
                        ):
                            agent_answer_element = agent_answer_element[1:]
                            head_strip_flag = True
                        if (
                            agent_answer_element.endswith(char)
                            and not ground_truth_element.endswith(char)
                            and not tail_strip_flag
                        ):
                            agent_answer_element = agent_answer_element[:-1]
                            tail_strip_flag = True
                    if agent_answer_element != ground_truth_element:
                        return False
                if agent_answer_element != ground_truth_element:
                    return False
                # endregion
            else:
                # This could never reach if DBBenchType.Row = list[tuple[int | str | float, ...]]
                # It is only kept for the correctness of the code
                return False
        # endregion
        return True

    @staticmethod
    def validate(agent_answer: str, ground_truth: DBBenchType.AnswerDirect) -> bool:
        if (
            tuple_position_list := DirectTypeAnswerValidator._get_tuple_position_list(
                agent_answer
            )
        ) is None:
            return False
        if len(tuple_position_list) == 0:
            if len(ground_truth) == 0:
                return True
            if len(ground_truth) != 1:
                return False
            return DirectTypeAnswerValidator._validate_single_tuple_str(
                agent_answer, ground_truth[0]
            )
        if len(tuple_position_list) != len(ground_truth):
            return False
        for tuple_position, ground_truth_tuple in zip(
            tuple_position_list, ground_truth
        ):
            start_index, end_index = tuple_position
            tuple_str = agent_answer[start_index + 1 : end_index]
            if not DirectTypeAnswerValidator._validate_single_tuple_str(
                tuple_str, ground_truth_tuple
            ):
                return False
        return True
