from math import ceil, floor
from queue import Queue
from typing import List

from .v_schedule import ScheduledNode

DUALPIPE_NODETYPE = {"F", "B", "W", "Full_B", "EMPTY_BUBBLE"}


class DualPipeGraph(object):
    """DualPipeGraph
    We brokendown DualPipe to three Pipe_Stage: Warmup, Middle, End
    Warmup contains:
        step1: no_cross_fwd
        step2: cross_fwd
        step3: warmup_1F1B1W
        step4: warmup_transitions
    Middle contains: (named as their shape in pipe)
        step1: mid_rhombic
        step2: mid_butterfly
        step3: mid_transitions
    End contains:
        step1: bwdB_step
        step2: cross_bwdB_bwdW
        step3: bwdW_step
    """

    def __init__(
        self,
        n_stage,
        n_micro,
        f_cost: int = 1,
        b_cost: int = 1,
        w_cost: int = 1,
        c_cost: int = 1,
        f_mem: int = 1,
        b_mem: int = 1,
        w_mem: int = 1,
        max_mem: int = None,
    ):
        self.n_node = 6 * n_stage * n_micro
        self.n_stage = n_stage
        self.n_micro = n_micro
        self.f_cost = f_cost
        self.b_cost = b_cost
        self.w_cost = w_cost
        self.c_cost = c_cost
        self.f_mem = f_mem
        self.b_mem = b_mem
        self.w_mem = w_mem
        self.fbw_cost = [f_cost, b_cost, w_cost]
        self.fbw_mem = [f_mem, b_mem, w_mem]
        self.max_mem = max_mem or f_mem * self.n_stage * 2

        # time unit
        self.one_time_unit = 1
        # total mbs (both up and down)
        self.total_mbs = (n_stage + 2) * 2
        # one side mbs
        self.mbs = self.total_mbs // 2

    def print_details(
        self,
        pipeline_schedule: List[List[ScheduledNode]],
        chunk_mode: bool = False,
        mbs_mode: bool = False,
        empty_bubble_str_mode: bool = False,
    ):
        assert not (
            chunk_mode and mbs_mode
        ), "Only one mode is supported at the same time, please choose from chunk_mode and mbs_mode"
        schedule_str = ""
        for stage in range(self.n_stage):
            stage_nodes = []
            for node in pipeline_schedule[stage]:
                if node.type in DUALPIPE_NODETYPE:
                    if node.type == "EMPTY_BUBBLE":
                        if empty_bubble_str_mode:
                            stage_nodes.append("E")
                        else:
                            stage_nodes.append(" ")
                    else:
                        if chunk_mode:
                            stage_nodes.append(node.type + str(node.chunk))
                        elif mbs_mode:
                            stage_nodes.append(node.type + str(node.minibatch))
                        else:
                            stage_nodes.append(node.type)
            stage_str = "".join([_ for _ in stage_nodes])
            schedule_str += "\n" + stage_str
        print(schedule_str)

    # def get_pipe_stage(self, stage_pipe: List[ScheduledNode]):
    #     # get first d, last d, first u, last u B node in range[first B, first W]
    #     cut_index = 0
    #     first_d, last_d, first_u, last_u = self.n_micro, 0, self.n_micro, 0
    #     for i in range(len(stage_pipe)):
    #         if stage_pipe[i].type ==  'Full_B':
    #             cut_index = i
    #             break
    #     stage_pipe = stage_pipe[cut_index:] # node from last fully B to ...

    #     for node in stage_pipe:
    #         if node.type ==  'B':
    #             first_d = node.minibatch + 1

    #     return 0, 0, 0, 0
    ################
    # Pipe_Stage 1
    ################
    def get_warmup_schedule(self, pipeline_schedule: List[List[ScheduledNode]]):
        ########### Pipe_Stage 1.1 ###########
        def no_cross_fwd(pipeline_schedule: List[List[ScheduledNode]]):
            # stage [0,pp/2)
            for stage in range(0, self.n_stage // 2):
                # add num i empty bubble
                start_time = 0
                for i in range(0, stage):
                    pipeline_schedule[stage].append(
                        ScheduledNode(
                            type="EMPTY_BUBBLE",
                            chunk=0,
                            stage=stage,
                            minibatch=0,
                            start_time=start_time,
                            completion_time=start_time + self.one_time_unit,
                        )
                    )
                    start_time += self.one_time_unit
                # add FWD node
                # Stage i in [0, pp/2)  mbs m in range [0, (pp - 1） - 2i) model chunk 0 fwd
                curr_time = pipeline_schedule[stage][-1].completion_time if pipeline_schedule[stage] else 0
                for i in range(0, (self.n_stage - 1) - 2 * stage):
                    pipeline_schedule[stage].append(
                        ScheduledNode(
                            type="F",
                            chunk=0,
                            stage=stage,
                            minibatch=i,
                            start_time=curr_time,
                            completion_time=curr_time + self.one_time_unit,
                        )
                    )
                    curr_time += self.one_time_unit

            # stage [pp/2,n_stage)
            for stage in range(self.n_stage // 2, self.n_stage):
                start_time = 0
                for i in range(0, self.n_stage - stage - 1):
                    # add num i empty bubble
                    pipeline_schedule[stage].append(
                        ScheduledNode(
                            type="EMPTY_BUBBLE",
                            chunk=0,
                            stage=stage,
                            minibatch=0,
                            start_time=start_time,
                            completion_time=start_time + self.one_time_unit,
                        )
                    )
                    start_time += self.one_time_unit
                # add FWD node
                # Stage i in [pp/2, pp) , mbs m in range [0, 2i - (pp-1)) model chunk 1 fwd
                curr_time = pipeline_schedule[stage][-1].completion_time if pipeline_schedule[stage] else 0
                for i in range(0, 2 * stage - (self.n_stage - 1)):
                    pipeline_schedule[stage].append(
                        ScheduledNode(
                            type="F",
                            chunk=1,
                            stage=stage,
                            minibatch=i,
                            start_time=curr_time,
                            completion_time=curr_time + self.one_time_unit,
                        )
                    )
                    curr_time += self.one_time_unit

        ########### Pipe_Stage 1.2 ###########
        # For each stage, add schedule Nodes col pp/2 times (range 0 to self.n_stage//2 + 1)，
        def cross_fwd(pipeline_schedule: List[List[ScheduledNode]]):
            for r in range(0, self.n_stage // 2 + 1):
                if r == 0:
                    # special case; only one col 0d 0u
                    for stage in range(r, self.n_stage // 2):
                        curr_time = pipeline_schedule[stage][-1].completion_time if pipeline_schedule[stage] else 0
                        pipeline_schedule[stage].append(
                            ScheduledNode(
                                type="F",
                                chunk=1,
                                stage=stage,
                                minibatch=0,
                                start_time=curr_time,
                                completion_time=curr_time + self.one_time_unit,
                            )
                        )
                        curr_time += self.one_time_unit

                    for stage in range(self.n_stage // 2, self.n_stage - r):
                        curr_time = pipeline_schedule[stage][-1].completion_time if pipeline_schedule[stage] else 0
                        pipeline_schedule[stage].append(
                            ScheduledNode(
                                type="F",
                                chunk=0,
                                stage=stage,
                                minibatch=0,
                                start_time=curr_time,
                                completion_time=curr_time + self.one_time_unit,
                            )
                        )
                        curr_time += self.one_time_unit
                else:
                    if r % 2 != 0:
                        for stage in range(r, self.n_stage // 2):
                            curr_time = pipeline_schedule[stage][-1].completion_time if pipeline_schedule[stage] else 0
                            # Stage i in [r, pp/2) , mbs为 (pp - 1） - 2i + (r-1) model chunk 0 fwd
                            pipeline_schedule[stage].append(
                                ScheduledNode(
                                    type="F",
                                    chunk=0,
                                    stage=stage,
                                    minibatch=(self.n_stage - 1) - 2 * stage + (r - 1),
                                    start_time=curr_time,
                                    completion_time=curr_time + self.one_time_unit,
                                )
                            )
                            curr_time += self.one_time_unit

                            # Stage i in [r, pp/2) , mbs r model chunk 1 fwd # 全1d
                            pipeline_schedule[stage].append(
                                ScheduledNode(
                                    type="F",
                                    chunk=1,
                                    stage=stage,
                                    minibatch=r,
                                    start_time=curr_time,
                                    completion_time=curr_time + self.one_time_unit,
                                )
                            )
                            curr_time += self.one_time_unit

                        for stage in range(self.n_stage // 2, self.n_stage - r):
                            curr_time = pipeline_schedule[stage][-1].completion_time if pipeline_schedule[stage] else 0
                            # Stage i in [pp/2, pp-r), mbs 为2i - (pp-1)  + (r-1) model chunk 1 fwd
                            pipeline_schedule[stage].append(
                                ScheduledNode(
                                    type="F",
                                    chunk=1,
                                    stage=stage,
                                    minibatch=2 * stage - (self.n_stage - 1) + (r - 1),
                                    start_time=curr_time,
                                    completion_time=curr_time + self.one_time_unit,
                                )
                            )
                            curr_time += self.one_time_unit

                            # Stage i in [pp/2, pp-r) 压入mbs r model chunk 0 fwd # 全1u
                            pipeline_schedule[stage].append(
                                ScheduledNode(
                                    type="F",
                                    chunk=0,
                                    stage=stage,
                                    minibatch=r,
                                    start_time=curr_time,
                                    completion_time=curr_time + self.one_time_unit,
                                )
                            )
                            curr_time += self.one_time_unit

                    if r % 2 == 0:
                        for stage in range(r, self.n_stage // 2):
                            curr_time = pipeline_schedule[stage][-1].completion_time if pipeline_schedule[stage] else 0
                            # Stage i in [r, pp/2) , mbs为 (pp - 1） - 2i + (r-1) model chunk 0 fwd
                            pipeline_schedule[stage].append(
                                ScheduledNode(
                                    type="F",
                                    chunk=0,
                                    stage=stage,
                                    minibatch=self.n_stage - 2 * stage + (r - 2),
                                    start_time=curr_time,
                                    completion_time=curr_time + self.one_time_unit,
                                )
                            )
                            curr_time += self.one_time_unit

                            # Stage i in [r, pp/2) , mbs r model chunk 1 fwd # 全1d
                            pipeline_schedule[stage].append(
                                ScheduledNode(
                                    type="F",
                                    chunk=1,
                                    stage=stage,
                                    minibatch=r,
                                    start_time=curr_time,
                                    completion_time=curr_time + self.one_time_unit,
                                )
                            )
                            curr_time += self.one_time_unit

                        for stage in range(self.n_stage // 2, self.n_stage - r):
                            curr_time = pipeline_schedule[stage][-1].completion_time if pipeline_schedule[stage] else 0
                            # Stage i in [pp/2, pp-r), mbs 为2i - (pp-1)  + (r-1) model chunk 1 fwd
                            pipeline_schedule[stage].append(
                                ScheduledNode(
                                    type="F",
                                    chunk=1,
                                    stage=stage,
                                    minibatch=2 * stage - (self.n_stage - 2) + (r - 2),
                                    start_time=curr_time,
                                    completion_time=curr_time + self.one_time_unit,
                                )
                            )
                            curr_time += self.one_time_unit

                            # Stage i in [pp/2, pp-r) 压入mbs r model chunk 0 fwd # 全1u
                            pipeline_schedule[stage].append(
                                ScheduledNode(
                                    type="F",
                                    chunk=0,
                                    stage=stage,
                                    minibatch=r,
                                    start_time=curr_time,
                                    completion_time=curr_time + self.one_time_unit,
                                )
                            )
                            curr_time += self.one_time_unit

        ########### Pipe_Stage 1.3 ###########
        def warmup_1f1b1w(pipeline_schedule: List[List[ScheduledNode]]):
            # for each stage, add Schedule Nodes pp/2 times from (0, self.n_stage//2 + 1)
            for r in range(0, self.n_stage // 2):
                # [0, pp/2 - r)
                for stage in range(0, self.n_stage // 2 - 1 - r):
                    ###### bwd b ######
                    curr_time = pipeline_schedule[stage][-1].completion_time if pipeline_schedule[stage] else 0
                    # Stage i in [r, pp/2) , mbs为 (pp - 1） - 2i + (r-1) model chunk 0 fwd
                    pipeline_schedule[stage].append(
                        ScheduledNode(
                            type="B",
                            chunk=1,
                            stage=stage,
                            minibatch=r,
                            start_time=curr_time,
                            completion_time=curr_time + self.one_time_unit,
                        )
                    )
                    curr_time += self.one_time_unit
                    ###### bwd w ######
                    # Stage i in [0, pp/2 - r) , mbs r model chunk 1  bwd w
                    pipeline_schedule[stage].append(
                        ScheduledNode(
                            type="W",
                            chunk=1,
                            stage=stage,
                            minibatch=r,
                            start_time=curr_time,
                            completion_time=curr_time + self.one_time_unit,
                        )
                    )
                    curr_time += self.one_time_unit
                    ###### fwd ######
                    # Stage i in [0, pp/2 - r), mbs i + 1 + r model chunk 1 fwd
                    pipeline_schedule[stage].append(
                        ScheduledNode(
                            type="F",
                            chunk=1,
                            stage=stage,
                            minibatch=stage + 1 + r,
                            start_time=curr_time,
                            completion_time=curr_time + self.one_time_unit,
                        )
                    )
                    curr_time += self.one_time_unit

                # [pp/2 + 1 + r, pp)
                for stage in range(self.n_stage // 2 + 1 + r, self.n_stage):
                    ###### bwd b ######
                    curr_time = pipeline_schedule[stage][-1].completion_time if pipeline_schedule[stage] else 0
                    # Stage i in [0, pp/2 - r), mbs i + 1 + r model chunk 1 fwd
                    pipeline_schedule[stage].append(
                        ScheduledNode(
                            type="B",
                            chunk=0,
                            stage=stage,
                            minibatch=r,
                            start_time=curr_time,
                            completion_time=curr_time + self.one_time_unit,
                        )
                    )
                    curr_time += self.one_time_unit
                    ###### bwd w ######
                    # Stage i in [pp/2 + 1 - r, pp) ,mbs r model chunk 0  bwd w
                    pipeline_schedule[stage].append(
                        ScheduledNode(
                            type="W",
                            chunk=0,
                            stage=stage,
                            minibatch=r,
                            start_time=curr_time,
                            completion_time=curr_time + self.one_time_unit,
                        )
                    )
                    curr_time += self.one_time_unit
                    ###### fwd ######
                    # Stage i in [0, pp/2 - r) , mbs i + 1 + r model chunk 1 fwd
                    pipeline_schedule[stage].append(
                        ScheduledNode(
                            type="F",
                            chunk=0,
                            stage=stage,
                            minibatch=self.n_stage - stage + r,
                            start_time=curr_time,
                            completion_time=curr_time + self.one_time_unit,
                        )
                    )
                    curr_time += self.one_time_unit

        ########### Pipe_Stage 1.4 ###########
        def warmup_transitions(pipeline_schedule: List[List[ScheduledNode]]):
            # For each stage, add pp/2 - 1 round Schedule Nodes
            for r in range(0, self.n_stage // 2):
                if r == 0:
                    # special round add 1 col fwd and 1 col fullyBwd
                    for stage in range(0, self.n_stage // 2):
                        ###### Fwd ######
                        curr_time = pipeline_schedule[stage][-1].completion_time if pipeline_schedule[stage] else 0
                        # Stage i in [0, pp/2 ) , mbs pp - i - 1 - r model chunk 0 fwd
                        pipeline_schedule[stage].append(
                            ScheduledNode(
                                type="F",
                                chunk=0,
                                stage=stage,
                                minibatch=self.n_stage - stage - 1 - r,
                                start_time=curr_time,
                                completion_time=curr_time + self.one_time_unit,
                            )
                        )
                        curr_time += self.one_time_unit
                        ###### Fully Bwd ######
                        # Stage i in [0, pp/2) , add mbs (pp/2) - i - 1 - r model chunk 0 Full_B
                        pipeline_schedule[stage].append(
                            ScheduledNode(
                                type="Full_B",
                                chunk=0,
                                stage=stage,
                                minibatch=self.n_stage // 2 - stage - 1 - r,
                                start_time=curr_time,
                                completion_time=curr_time + self.one_time_unit,
                            )
                        )
                        curr_time += self.one_time_unit
                    for stage in range(self.n_stage // 2, self.n_stage):
                        ###### Fwd ######
                        curr_time = pipeline_schedule[stage][-1].completion_time if pipeline_schedule[stage] else 0
                        # Stage i in [pp/2, pp), mbs i + r model chunk chunk 1  fwd
                        pipeline_schedule[stage].append(
                            ScheduledNode(
                                type="F",
                                chunk=1,
                                stage=stage,
                                minibatch=stage + r,
                                start_time=curr_time,
                                completion_time=curr_time + self.one_time_unit,
                            )
                        )
                        curr_time += self.one_time_unit
                        ###### Fully Bwd ######
                        # Stage i in [pp/2, pp), mbs i - (pp/2) + r model chunk chunk 1 FullyB
                        pipeline_schedule[stage].append(
                            ScheduledNode(
                                type="Full_B",
                                chunk=1,
                                stage=stage,
                                minibatch=stage - self.n_stage // 2 + r,
                                start_time=curr_time,
                                completion_time=curr_time + self.one_time_unit,
                            )
                        )
                        curr_time += self.one_time_unit
                else:
                    for stage in range(0, self.n_stage // 2 - r):
                        curr_time = pipeline_schedule[stage][-1].completion_time if pipeline_schedule[stage] else 0
                        # Stage i in [0, pp/2 - r) EMPTY_BUBBLE
                        pipeline_schedule[stage].append(
                            ScheduledNode(
                                type="EMPTY_BUBBLE",
                                chunk=0,
                                stage=stage,
                                minibatch=0,
                                start_time=curr_time,
                                completion_time=curr_time + self.one_time_unit,
                            )
                        )
                        curr_time += self.one_time_unit
                    for stage in range(self.n_stage // 2 + r, self.n_stage):
                        curr_time = pipeline_schedule[stage][-1].completion_time if pipeline_schedule[stage] else 0
                        # Stage i in [pp/2 + r, pp) 压入空泡
                        pipeline_schedule[stage].append(
                            ScheduledNode(
                                type="EMPTY_BUBBLE",
                                chunk=0,
                                stage=stage,
                                minibatch=0,
                                start_time=curr_time,
                                completion_time=curr_time + self.one_time_unit,
                            )
                        )
                        curr_time += self.one_time_unit

        no_cross_fwd(pipeline_schedule)
        cross_fwd(pipeline_schedule)
        warmup_1f1b1w(pipeline_schedule)
        warmup_transitions(pipeline_schedule)

    ################
    # Pipe_Stage 2
    ################
    def get_middle_schedule(self, pipeline_schedule: List[List[ScheduledNode]]):
        ########### Pipe_Stage 2.1 ###########
        def mid_rhombic(pipeline_schedule: List[List[ScheduledNode]]):
            # for each stage, add (pp/2) + 1 round（total 9 round）Schedule Nodes
            for r in range(0, self.n_stage // 2 + 1):
                if r == 0:
                    for stage in range(0, self.n_stage // 2):
                        curr_time = pipeline_schedule[stage][-1].completion_time if pipeline_schedule[stage] else 0
                        ###### Fwd 1 ######
                        # Stage i in [0, pp/2 ), mbs pp/2 model chunk 1 fwd
                        pipeline_schedule[stage].append(
                            ScheduledNode(
                                type="F",
                                chunk=1,
                                stage=stage,
                                minibatch=self.n_stage // 2 + r,
                                start_time=curr_time,
                                completion_time=curr_time + self.one_time_unit,
                            )
                        )
                        curr_time += self.one_time_unit
                        ###### Fully Bwd 1 ######
                        # Stage i in [0, pp/2) , mbs r model chunk 0 Fully Bwd
                        pipeline_schedule[stage].append(
                            ScheduledNode(
                                type="Full_B",
                                chunk=0,
                                stage=stage,
                                minibatch=r,
                                start_time=curr_time,
                                completion_time=curr_time + self.one_time_unit,
                            )
                        )
                        curr_time += self.one_time_unit

                        ###### Fwd 2 ######
                        # Stage i in [0, pp/2 ) , mbs pp - i -r model chunk 0 fwd
                        pipeline_schedule[stage].append(
                            ScheduledNode(
                                type="F",
                                chunk=0,
                                stage=stage,
                                minibatch=self.n_stage - stage - r,
                                start_time=curr_time,
                                completion_time=curr_time + self.one_time_unit,
                            )
                        )
                        curr_time += self.one_time_unit

                        ###### Fully Bwd 2 ######
                        # Stage i in [0, pp/2) , mbs (pp/2) - i - r model chunk 1 fwd
                        pipeline_schedule[stage].append(
                            ScheduledNode(
                                type="Full_B",
                                chunk=1,
                                stage=stage,
                                minibatch=self.n_stage // 2 - stage - r,
                                start_time=curr_time,
                                completion_time=curr_time + self.one_time_unit,
                            )
                        )
                        curr_time += self.one_time_unit
                    for stage in range(self.n_stage // 2, self.n_stage):
                        curr_time = pipeline_schedule[stage][-1].completion_time if pipeline_schedule[stage] else 0
                        ###### Fwd 1 ######
                        # Stage i in [pp/2, pp), mbs pp/2 + r model chunk chunk 0 fwd
                        pipeline_schedule[stage].append(
                            ScheduledNode(
                                type="F",
                                chunk=0,
                                stage=stage,
                                minibatch=self.n_stage // 2 + r,
                                start_time=curr_time,
                                completion_time=curr_time + self.one_time_unit,
                            )
                        )
                        curr_time += self.one_time_unit
                        ###### Fully Bwd 1 ######
                        # Stage i in [0, pp/2) , mbs r model chunk 0 Fully Bwd
                        pipeline_schedule[stage].append(
                            ScheduledNode(
                                type="Full_B",
                                chunk=1,
                                stage=stage,
                                minibatch=r,
                                start_time=curr_time,
                                completion_time=curr_time + self.one_time_unit,
                            )
                        )
                        curr_time += self.one_time_unit

                        ###### Fwd 2 ######
                        # Stage i in [pp/2, pp) , mbs i + 1 + r model chunk 1 fwd
                        pipeline_schedule[stage].append(
                            ScheduledNode(
                                type="F",
                                chunk=1,
                                stage=stage,
                                minibatch=stage + 1 + r,
                                start_time=curr_time,
                                completion_time=curr_time + self.one_time_unit,
                            )
                        )
                        curr_time += self.one_time_unit
                        ###### Fully Bwd 2 ######
                        # Stage i in [pp/2, pp) , mbs i - ((pp/2) -1) + r model chunk chunk 0 Fully Bwd
                        pipeline_schedule[stage].append(
                            ScheduledNode(
                                type="Full_B",
                                chunk=0,
                                stage=stage,
                                minibatch=stage - (self.n_stage // 2 - 1) + r,
                                start_time=curr_time,
                                completion_time=curr_time + self.one_time_unit,
                            )
                        )
                        curr_time += self.one_time_unit
                else:
                    for stage in range(r - 1, self.n_stage // 2):
                        curr_time = pipeline_schedule[stage][-1].completion_time if pipeline_schedule[stage] else 0
                        ###### Fwd 1 ######
                        # Stage i in [r - 1, pp/2), mbs (pp/2) + r model chunk 1 fwd
                        pipeline_schedule[stage].append(
                            ScheduledNode(
                                type="F",
                                chunk=1,
                                stage=stage,
                                minibatch=self.n_stage // 2 + r,
                                start_time=curr_time,
                                completion_time=curr_time + self.one_time_unit,
                            )
                        )
                        curr_time += self.one_time_unit

                        ###### Fully Bwd 1 ######
                        # Stage i in [r - 1, pp/2), mbs r model chunk 0 Fully Bwd
                        pipeline_schedule[stage].append(
                            ScheduledNode(
                                type="Full_B",
                                chunk=0,
                                stage=stage,
                                minibatch=r,
                                start_time=curr_time,
                                completion_time=curr_time + self.one_time_unit,
                            )
                        )
                        curr_time += self.one_time_unit

                        ###### Fwd 2 ######
                        # Stage i in [r - 1, pp/2) , mbs pp + r - i model chunk 0 fwd
                        pipeline_schedule[stage].append(
                            ScheduledNode(
                                type="F",
                                chunk=0,
                                stage=stage,
                                minibatch=self.n_stage + r - stage,
                                start_time=curr_time,
                                completion_time=curr_time + self.one_time_unit,
                            )
                        )
                        curr_time += self.one_time_unit

                        ###### Fully Bwd 2 ######
                        # Stage i in [r - 1, pp/2), mbs (pp/2) - i + r model chunk 1 Fully Bwd
                        pipeline_schedule[stage].append(
                            ScheduledNode(
                                type="Full_B",
                                chunk=1,
                                stage=stage,
                                minibatch=self.n_stage // 2 - stage + r,
                                start_time=curr_time,
                                completion_time=curr_time + self.one_time_unit,
                            )
                        )
                        curr_time += self.one_time_unit
                    for stage in range(self.n_stage // 2, self.n_stage - r + 1):
                        curr_time = pipeline_schedule[stage][-1].completion_time if pipeline_schedule[stage] else 0
                        ###### Fwd 1 ######
                        # Stage i in [pp/2, pp - r + 1) , mbs (pp/2) + r model chunk chunk 0 fwd
                        pipeline_schedule[stage].append(
                            ScheduledNode(
                                type="F",
                                chunk=0,
                                stage=stage,
                                minibatch=self.n_stage // 2 + r,
                                start_time=curr_time,
                                completion_time=curr_time + self.one_time_unit,
                            )
                        )
                        curr_time += self.one_time_unit

                        ###### Fully Bwd 1 ######
                        # Stage i in [pp/2, pp - r + 1), mbs r model chunk chunk 1 Fully Bwd
                        pipeline_schedule[stage].append(
                            ScheduledNode(
                                type="Full_B",
                                chunk=1,
                                stage=stage,
                                minibatch=r,
                                start_time=curr_time,
                                completion_time=curr_time + self.one_time_unit,
                            )
                        )
                        curr_time += self.one_time_unit

                        ###### Fwd 2 ######
                        # Stage i in [pp/2, pp - r + 1), mbs i + 1 + r model chunk chunk 1 fwd
                        pipeline_schedule[stage].append(
                            ScheduledNode(
                                type="F",
                                chunk=1,
                                stage=stage,
                                minibatch=stage + 1 + r,
                                start_time=curr_time,
                                completion_time=curr_time + self.one_time_unit,
                            )
                        )
                        curr_time += self.one_time_unit
                        ###### Fully Bwd 2 ######
                        # Stage i in [pp/2, pp - r + 1), mbs i - ((pp/2) -1) + r model chunk chunk 0 Fully Bwd
                        pipeline_schedule[stage].append(
                            ScheduledNode(
                                type="Full_B",
                                chunk=0,
                                stage=stage,
                                minibatch=stage - (self.n_stage // 2 - 1) + r,
                                start_time=curr_time,
                                completion_time=curr_time + self.one_time_unit,
                            )
                        )
                        curr_time += self.one_time_unit

        ########### Pipe_Stage 2.2 ###########
        def mid_butterfly(pipeline_schedule: List[List[ScheduledNode]]):
            # for each stage, add pp/2 round（total 8 round）Schedule Nodes
            for r in range(0, self.n_stage // 2 + 1):
                if r == 0:
                    for stage in range(0, self.n_stage // 2 - r):
                        curr_time = pipeline_schedule[stage][-1].completion_time if pipeline_schedule[stage] else 0
                        ###### Fwd ######
                        # Stage i in [0, pp/2 - r ), mbs bs//2-pp//2+i+r model chunk 1 fwd
                        pipeline_schedule[stage].append(
                            ScheduledNode(
                                type="F",
                                chunk=1,
                                stage=stage,
                                minibatch=self.n_micro // 2 - self.n_stage // 2 + stage + r,
                                start_time=curr_time,
                                completion_time=curr_time + self.one_time_unit,
                            )
                        )
                        curr_time += self.one_time_unit
                    for stage in range(self.n_stage // 2 + r, self.n_stage):
                        curr_time = pipeline_schedule[stage][-1].completion_time if pipeline_schedule[stage] else 0
                        ###### Fwd ######
                        # Stage i in [pp/2 + r, pp), mbs (bs//2 + pp//2 - 1)- i + r model chunk chunk 0 fwd
                        pipeline_schedule[stage].append(
                            ScheduledNode(
                                type="F",
                                chunk=0,
                                stage=stage,
                                minibatch=self.n_micro // 2 + self.n_stage // 2 - 1 - stage + r,
                                start_time=curr_time,
                                completion_time=curr_time + self.one_time_unit,
                            )
                        )
                        curr_time += self.one_time_unit
                else:
                    for stage in range(0, self.n_stage // 2 - r):
                        curr_time = pipeline_schedule[stage][-1].completion_time if pipeline_schedule[stage] else 0
                        ###### Fully bwd 1 ######
                        # Stage i in [0, pp/2 - r ), mbs i + r + 1 model chunk 0 fwd
                        pipeline_schedule[stage].append(
                            ScheduledNode(
                                type="Full_B",
                                chunk=0,
                                stage=stage,
                                minibatch=stage + r + 1,
                                start_time=curr_time,
                                completion_time=curr_time + self.one_time_unit,
                            )
                        )
                        curr_time += self.one_time_unit
                        ###### Empty Bubble ######
                        # Stage i in [0, pp/2 - r) , Empty Bubble
                        pipeline_schedule[stage].append(
                            ScheduledNode(
                                type="EMPTY_BUBBLE",
                                chunk=0,
                                stage=stage,
                                minibatch=0,
                                start_time=curr_time,
                                completion_time=curr_time + self.one_time_unit,
                            )
                        )
                        curr_time += self.one_time_unit

                        ###### Fully bwd 2 ######
                        # Stage i in [0, pp/2 - r), mbs bs//2-pp//2 - 1 + r model chunk 1 Fully bwd
                        pipeline_schedule[stage].append(
                            ScheduledNode(
                                type="Full_B",
                                chunk=1,
                                stage=stage,
                                minibatch=self.n_micro // 2 - self.n_stage // 2 - 1 + r,
                                start_time=curr_time,
                                completion_time=curr_time + self.one_time_unit,
                            )
                        )
                        curr_time += self.one_time_unit

                        ###### Fwd ######
                        # Stage i in [0, pp/2 - r), mbs bs//2 - pp//2 + i + r model chunk 1 fwd
                        pipeline_schedule[stage].append(
                            ScheduledNode(
                                type="F",
                                chunk=1,
                                stage=stage,
                                minibatch=self.n_micro // 2 - self.n_stage // 2 + stage + r,
                                start_time=curr_time,
                                completion_time=curr_time + self.one_time_unit,
                            )
                        )
                        curr_time += self.one_time_unit

                    for stage in range(self.n_stage // 2 + r, self.n_stage):
                        curr_time = pipeline_schedule[stage][-1].completion_time if pipeline_schedule[stage] else 0
                        ###### Fully bwd 1 ######
                        # Stage i in [pp/2 + r, pp), mbs bs - i + r -2 model chunk 1 fwd
                        pipeline_schedule[stage].append(
                            ScheduledNode(
                                type="Full_B",
                                chunk=1,
                                stage=stage,
                                minibatch=self.n_stage - stage + r,
                                start_time=curr_time,
                                completion_time=curr_time + self.one_time_unit,
                            )
                        )
                        curr_time += self.one_time_unit

                        ###### Empty Bubble ######
                        # Stage i in [pp/2 + r, pp), Empty Bubble
                        pipeline_schedule[stage].append(
                            ScheduledNode(
                                type="EMPTY_BUBBLE",
                                chunk=0,
                                stage=stage,
                                minibatch=0,
                                start_time=curr_time,
                                completion_time=curr_time + self.one_time_unit,
                            )
                        )
                        curr_time += self.one_time_unit
                        ###### Fully bwd 2 ######
                        # Stage i in [pp/2 + r, pp), mbs bs//2-pp//2 - 1 + r model chunk 0 Fully bwd
                        pipeline_schedule[stage].append(
                            ScheduledNode(
                                type="Full_B",
                                chunk=1,
                                stage=stage,
                                minibatch=self.n_micro // 2 - self.n_stage // 2 - 1 + r,
                                start_time=curr_time,
                                completion_time=curr_time + self.one_time_unit,
                            )
                        )
                        curr_time += self.one_time_unit

                        ###### Fwd ######
                        # Stage i in [pp/2 + r, pp), mbs (bs/2+pp/2 - 1) - i + r model chunk 0 fwd
                        pipeline_schedule[stage].append(
                            ScheduledNode(
                                type="F",
                                chunk=0,
                                stage=stage,
                                minibatch=self.n_micro // 2 + self.n_stage // 2 - 1 - stage + r,
                                start_time=curr_time,
                                completion_time=curr_time + self.one_time_unit,
                            )
                        )
                        curr_time += self.one_time_unit

        ########### Pipe_Stage 2.3 ###########
        def mid_transitions(pipeline_schedule: List[List[ScheduledNode]]):
            # for each stage, add pp/2 + 1 round（total 9 round）Schedule Nodes
            for r in range(0, self.n_stage // 2 + 1):
                if r == 0:
                    for stage in range(r, self.n_stage // 2):
                        ###### Fully B ######
                        # Stage i in [r, pp/2), mbs pp/2 + r + 1  model chunk 0 fully bwd
                        curr_time = pipeline_schedule[stage][-1].completion_time if pipeline_schedule[stage] else 0
                        pipeline_schedule[stage].append(
                            ScheduledNode(
                                type="Full_B",
                                chunk=0,
                                stage=stage,
                                minibatch=self.n_stage // 2 + r + 1,
                                start_time=curr_time,
                                completion_time=curr_time + self.one_time_unit,
                            )
                        )
                        curr_time += self.one_time_unit

                    for stage in range(self.n_stage // 2, self.n_stage):
                        ###### Fully B ######
                        # Stage i in [pp/2, pp), mbs pp/2 + r + 1 model chunk 1 fully bwd
                        curr_time = pipeline_schedule[stage][-1].completion_time if pipeline_schedule[stage] else 0
                        pipeline_schedule[stage].append(
                            ScheduledNode(
                                type="Full_B",
                                chunk=1,
                                stage=stage,
                                minibatch=self.n_stage // 2 + r + 1,
                                start_time=curr_time,
                                completion_time=curr_time + self.one_time_unit,
                            )
                        )
                        curr_time += self.one_time_unit
                else:
                    if r % 2 != 0:  # odd round: 1, 3, 5, 7
                        for stage in range(r - 1, self.n_stage // 2):
                            curr_time = pipeline_schedule[stage][-1].completion_time if pipeline_schedule[stage] else 0
                            ###### Empty Bubble ######
                            # Stage i in [r - 1, pp/2), Empty bubble
                            pipeline_schedule[stage].append(
                                ScheduledNode(
                                    type="EMPTY_BUBBLE",
                                    chunk=0,
                                    stage=stage,
                                    minibatch=0,
                                    start_time=curr_time,
                                    completion_time=curr_time + self.one_time_unit,
                                )
                            )
                            curr_time += self.one_time_unit

                            ###### Fully bwd ######
                            # Stage i in [r - 1, pp/2), mbs pp + r - i model chunk 1 fully bwd
                            pipeline_schedule[stage].append(
                                ScheduledNode(
                                    type="Full_B",
                                    chunk=1,
                                    stage=stage,
                                    minibatch=self.n_stage + ceil(r / 2) - stage,
                                    start_time=curr_time,
                                    completion_time=curr_time + self.one_time_unit,
                                )
                            )
                            curr_time += self.one_time_unit

                        for stage in range(self.n_stage // 2, self.n_stage - r + 1):
                            curr_time = pipeline_schedule[stage][-1].completion_time if pipeline_schedule[stage] else 0
                            ###### Empty Bubble ######
                            # Stage i in [pp/2, pp - r + 1), Empty bubble
                            pipeline_schedule[stage].append(
                                ScheduledNode(
                                    type="EMPTY_BUBBLE",
                                    chunk=0,
                                    stage=stage,
                                    minibatch=0,
                                    start_time=curr_time,
                                    completion_time=curr_time + self.one_time_unit,
                                )
                            )
                            curr_time += self.one_time_unit

                            ###### Fully bwd ######
                            # Stage i in [pp/2, pp - r + 1)  压入mbs i + 1 + r model chunk chunk 0  fully bwd
                            pipeline_schedule[stage].append(
                                ScheduledNode(
                                    type="Full_B",
                                    chunk=0,
                                    stage=stage,
                                    minibatch=stage + 1 + ceil(r / 2),
                                    start_time=curr_time,
                                    completion_time=curr_time + self.one_time_unit,
                                )
                            )
                            curr_time += self.one_time_unit

                    else:  # even round: 2, 4, 6, 8
                        for stage in range(r - 1, self.n_stage // 2):
                            curr_time = pipeline_schedule[stage][-1].completion_time if pipeline_schedule[stage] else 0
                            ###### Empty Bubble ######
                            pipeline_schedule[stage].append(
                                ScheduledNode(
                                    type="EMPTY_BUBBLE",
                                    chunk=0,
                                    stage=stage,
                                    minibatch=0,
                                    start_time=curr_time,
                                    completion_time=curr_time + self.one_time_unit,
                                )
                            )
                            curr_time += self.one_time_unit

                            ###### Fully bwd ######
                            # Stage i in [r - 1, pp/2), mbs pp//2 + r model chunk 0 fully bwd
                            pipeline_schedule[stage].append(
                                ScheduledNode(
                                    type="Full_B",
                                    chunk=0,
                                    stage=stage,
                                    minibatch=self.n_stage // 2 + floor(r / 2) + 1,
                                    start_time=curr_time,
                                    completion_time=curr_time + self.one_time_unit,
                                )
                            )
                            curr_time += self.one_time_unit

                        for stage in range(self.n_stage // 2, self.n_stage - r + 1):
                            curr_time = pipeline_schedule[stage][-1].completion_time if pipeline_schedule[stage] else 0
                            ###### Empty Bubble ######
                            pipeline_schedule[stage].append(
                                ScheduledNode(
                                    type="EMPTY_BUBBLE",
                                    chunk=0,
                                    stage=stage,
                                    minibatch=0,
                                    start_time=curr_time,
                                    completion_time=curr_time + self.one_time_unit,
                                )
                            )
                            curr_time += self.one_time_unit

                            ###### Fully bwd ######
                            # Stage i in [pp/2, pp - r + 1), mbs  pp//2 + r model chunk chunk 1 fully bwd
                            pipeline_schedule[stage].append(
                                ScheduledNode(
                                    type="Full_B",
                                    chunk=1,
                                    stage=stage,
                                    minibatch=self.n_stage // 2 + floor(r / 2) + 1,
                                    start_time=curr_time,
                                    completion_time=curr_time + self.one_time_unit,
                                )
                            )
                            curr_time += self.one_time_unit

        mid_rhombic(pipeline_schedule)
        mid_butterfly(pipeline_schedule)
        mid_transitions(pipeline_schedule)

    ################
    # Pipe_Stage 3
    ################
    def get_end_schedule(self, pipeline_schedule: List[List[ScheduledNode]]):

        ########### Pipe_Stage 3.1 ###########
        def bwdB_step(pipeline_schedule: List[List[ScheduledNode]]):
            # for each stage, pp/2 round（total 8 round）Schedule Nodes，
            for r in range(0, self.n_stage // 2):
                if r % 2 == 0:
                    # Stage i in [pp/2 - r - 1, pp/2)
                    for stage in range(self.n_stage // 2 - r - 1, self.n_stage // 2):
                        # Stage i in [pp/2 - r - 1, pp/2), mbs (pp/2 - 1）*2 + r/2 model chunk 1  bwd B
                        curr_time = pipeline_schedule[stage][-1].completion_time if pipeline_schedule[stage] else 0
                        pipeline_schedule[stage].append(
                            ScheduledNode(
                                type="B",
                                chunk=1,
                                stage=stage,
                                minibatch=(self.n_stage // 2 - 1) * 2 + r // 2,
                                start_time=curr_time,
                                completion_time=curr_time + self.one_time_unit,
                            )
                        )
                        curr_time += self.one_time_unit
                    # Stage i in [pp/2, pp/2 + r + 1)
                    for stage in range(self.n_stage // 2, self.n_stage // 2 + r + 1):
                        # Stage i in [pp/2, pp/2 + r+ 1), mbs (pp/2 - 1）*2 + r/2 model chunk 0 bwd B
                        curr_time = pipeline_schedule[stage][-1].completion_time if pipeline_schedule[stage] else 0
                        pipeline_schedule[stage].append(
                            ScheduledNode(
                                type="B",
                                chunk=0,
                                stage=stage,
                                minibatch=(self.n_stage // 2 - 1) * 2 + r // 2,
                                start_time=curr_time,
                                completion_time=curr_time + self.one_time_unit,
                            )
                        )
                        curr_time += self.one_time_unit
                else:
                    # Stage i in [pp/2 - r - 1, pp/2)
                    for stage in range(self.n_stage // 2 - r - 1, self.n_stage // 2):
                        # Stage i in [pp/2 - r - 1, pp/2), mbs pp-1-(pp/2 - i)+ floor(r/2)  model chunk 0 bwd B  # [6:13,7:14]
                        curr_time = pipeline_schedule[stage][-1].completion_time if pipeline_schedule[stage] else 0
                        pipeline_schedule[stage].append(
                            ScheduledNode(
                                type="B",
                                chunk=0,
                                stage=stage,
                                minibatch=self.n_stage - 1 - (self.n_stage // 2 - stage) + floor(r / 2),
                                start_time=curr_time,
                                completion_time=curr_time + self.one_time_unit,
                            )
                        )
                        curr_time += self.one_time_unit
                    # Stage i in [pp/2, pp/2 + r+ 1)
                    for stage in range(self.n_stage // 2, self.n_stage // 2 + r + 1):
                        # Stage i in [pp/2, pp/2 + r+ 1), mbs pp-1-(i - pp/2 + 1) + floor(r/2) model chunk 1 bwd B # [8:14,9:13]
                        curr_time = pipeline_schedule[stage][-1].completion_time if pipeline_schedule[stage] else 0
                        pipeline_schedule[stage].append(
                            ScheduledNode(
                                type="B",
                                chunk=1,
                                stage=stage,
                                minibatch=self.n_stage - 1 - (stage - self.n_stage // 2 + 1) + floor(r / 2),
                                start_time=curr_time,
                                completion_time=curr_time + self.one_time_unit,
                            )
                        )
                        curr_time += self.one_time_unit

        ########### Pipe_Stage 3.2 ###########
        def cross_bwdB_bwdW(pipeline_schedule: List[List[ScheduledNode]]):

            for stage in range(0, self.n_stage // 2):
                B_queue, W_queue = Queue(), Queue()
                last_down_of_B, last_up_of_B = self.n_micro, self.n_micro

            for stage in range(self.n_stage // 2, self.n_stage):
                pass

            # for r in range(0, self.n_stage // 2 - 1):

            # # Stage i in [0, pp/2 - r - 1)
            # for stage in range(0, self.n_stage // 2 - r - 1):
            #     ###### Bwd W ######

            #     ###### Bwd B ######
            #     # Stage i in [0, pp/2 - r - 1), mbs  bs/2-pp/2+1+i+r  model chunk 0 bwd B
            #     curr_time = pipeline_schedule[stage][-1].completion_time if pipeline_schedule[stage] else 0
            #     pipeline_schedule[stage].append(
            #         ScheduledNode(
            #             type="B",
            #             chunk=0,
            #             stage=stage,
            #             minibatch=self.n_micro // 2 - self.n_stage // 2 + 1 + stage + r,
            #             start_time=curr_time,
            #             completion_time=curr_time + self.one_time_unit,
            #         )
            #     )
            #     curr_time += self.one_time_unit
            #     pass

            # # Stage i in [pp/2 + r + 1, pp)
            # for stage in range(self.n_stage // 2 + r + 1, self.n_stage):
            #     ###### Bwd W ######

            #     ###### Bwd B ######
            #     # Stage i in [pp/2 + r + 1, pp)  压入mbs (bs/2+pp/2)-i+r model chunk chunk 1 bwd B
            #     curr_time = pipeline_schedule[stage][-1].completion_time if pipeline_schedule[stage] else 0
            #     pipeline_schedule[stage].append(
            #         ScheduledNode(
            #             type="B",
            #             chunk=1,
            #             stage=stage,
            #             minibatch=self.n_micro // 2 + self.n_stage // 2 - stage + r,
            #             start_time=curr_time,
            #             completion_time=curr_time + self.one_time_unit,
            #         )
            #     )
            #     curr_time += self.one_time_unit

        ########### Pipe_Stage 3.3 ###########
        def bwdW_step(pipeline_schedule: List[List[ScheduledNode]]):
            # for each stage, add pp/2 round（total 8 round）Schedule Nodes
            for r in range(0, self.n_stage // 2):
                for stage in range(0, self.n_stage // 2):
                    # Red up # [0, 7]
                    if stage in range(self.n_stage // 2 - r - 1, self.n_stage // 2 - 1):
                        # Stage i in [pp/2 - r, pp/2 - 1), mbs(pp/2 - 1）*2 + 向下取整(r/2) + 1 model chunk Chunk_num   bwd W  # None
                        # mbs_num = (pp/2 - 1）*2 + 向下取整(r/2)  if r < pp //2 // 2 else  (pp/2 - 1）*2 + (r -  pp //2 // 2)
                        # chunk_num   = 1 if r < pp //2 else 0
                        curr_time = pipeline_schedule[stage][-1].completion_time if pipeline_schedule[stage] else 0
                        chunk_num = 1 if r < self.n_stage // 2 // 2 else 0
                        mbs_num = (
                            (self.n_stage // 2 - 1) * 2 + r
                            if r < self.n_stage // 2 // 2
                            else (self.n_stage // 2 - 1) * 2 + (r - self.n_stage // 2 // 2)
                        )
                        pipeline_schedule[stage].append(
                            ScheduledNode(
                                type="W",
                                chunk=chunk_num,
                                stage=stage,
                                minibatch=mbs_num,
                                start_time=curr_time,
                                completion_time=curr_time + self.one_time_unit,
                            )
                        )
                        curr_time += self.one_time_unit
                    # Blue up [7, 8]
                    if stage in range(self.n_stage // 2 - 1, self.n_stage // 2):
                        # Stage i in [pp/2 - 1, pp/2), mbs (pp/2 - 1）*2 + floor(r/2) model chunk 1  bwd W
                        curr_time = pipeline_schedule[stage][-1].completion_time if pipeline_schedule[stage] else 0
                        pipeline_schedule[stage].append(
                            ScheduledNode(
                                type="W",
                                chunk=1 if r % 2 == 0 else 0,
                                stage=stage,
                                minibatch=(self.n_stage // 2 - 1) * 2 + floor(r / 2),
                                start_time=curr_time,
                                completion_time=curr_time + self.one_time_unit,
                            )
                        )
                        curr_time += self.one_time_unit

                for stage in range(self.n_stage // 2, self.n_stage):
                    # Blue down [8, 9]
                    if stage in range(self.n_stage // 2, self.n_stage // 2 + 1):
                        # Stage i in [pp/2, pp/2 + 1), mbs (pp/2 - 1）*2 + 向下取整(r/2) - 1 model chunk chunk 0  bwd W
                        curr_time = pipeline_schedule[stage][-1].completion_time if pipeline_schedule[stage] else 0
                        pipeline_schedule[stage].append(
                            ScheduledNode(
                                type="W",
                                chunk=0 if r % 2 == 0 else 1,
                                stage=stage,
                                minibatch=(self.n_stage // 2 - 1) * 2 + floor(r / 2),
                                start_time=curr_time,
                                completion_time=curr_time + self.one_time_unit,
                            )
                        )
                        curr_time += self.one_time_unit

                    # Red Down # [9, 16]
                    if stage in range(self.n_stage // 2 + 1, self.n_stage // 2 + r + 1):
                        # Stage i in [pp/2 + 1, pp/2 + r+ 1), mbs(pp/2 - 1）*2 + floor(r/2)  model chunk  0  bwd W
                        # mbs_num = (pp/2 - 1）*2 + 向下取整(r/2)  if r < pp //2 // 2 else  (pp/2 - 1）*2 + (r -  pp //2 // 2)
                        # chunk_num   = 1 if r < pp //2 else 0
                        curr_time = pipeline_schedule[stage][-1].completion_time if pipeline_schedule[stage] else 0
                        chunk_num = 0 if r < self.n_stage // 2 // 2 else 1
                        mbs_num = (
                            (self.n_stage // 2 - 1) * 2 + r
                            if r < self.n_stage // 2 // 2
                            else (self.n_stage // 2 - 1) * 2 + (r - self.n_stage // 2 // 2)
                        )
                        pipeline_schedule[stage].append(
                            ScheduledNode(
                                type="W",
                                chunk=chunk_num,
                                stage=stage,
                                minibatch=mbs_num,
                                start_time=curr_time,
                                completion_time=curr_time + self.one_time_unit,
                            )
                        )
                        curr_time += self.one_time_unit

        bwdB_step(pipeline_schedule)
        cross_bwdB_bwdW(pipeline_schedule)
        bwdW_step(pipeline_schedule)

    def get_dualpipe_schedule(
        self,
    ):
        pipeline_schedule = [[] for _ in range(self.n_stage)]
        self.get_warmup_schedule(pipeline_schedule)
        self.get_middle_schedule(pipeline_schedule)
        self.get_end_schedule(pipeline_schedule)
        return pipeline_schedule
