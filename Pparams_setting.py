IntervalDim = 100
VelocityDim = 32
VelocityOffset = IntervalDim

NoteOnDim = NoteOffDim = 128
NoteOnOffset = IntervalDim + VelocityDim
NoteOffOffset = IntervalDim + VelocityDim + NoteOnDim

CCDim = 2
CCOffset = IntervalDim + VelocityDim + NoteOnDim + NoteOffDim
EventDim = IntervalDim + VelocityDim + NoteOnDim + NoteOffDim + CCDim # 390

Time = 256
EmbeddingDim = 512
HeadDim = 16
Heads = 16
ContextDim = HeadDim * Heads # 512
Layers = 8

def default_hparams():
    return {
        'EventDim': EventDim,
        "ContextDim": ContextDim,
        'EmbeddingDim': EmbeddingDim,
        'Heads': Heads,
        'Layers': Layers,
        'Time': Time
    }

Batch = 1