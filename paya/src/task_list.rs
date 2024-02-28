use crate::command_recorder::CommandRecorder;

pub struct TaskList {
    tasks: Vec<Task>,
}

pub struct Task {
    pub name: String,
    pub task: Box<dyn FnMut()>,
}

pub struct FrameContext {
    pub frame_index: u64,
    pub command_recorder: CommandRecorder,
}

impl TaskList {
    pub fn new() -> Self {
        TaskList { tasks: Vec::new() }
    }

    pub fn add_task(&mut self, task: Task) {
        self.tasks.push(task);
    }
}
