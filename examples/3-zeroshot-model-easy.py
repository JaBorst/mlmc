##########################################################################
#
# Example for simply using a NLI-based zero-shot model
#
##########################################################################
import mlmc

m = mlmc.models.zeroshot.Encoder(
    classes={"Sports":0, "Science":1},
)

print(m.predict_batch(["Did you watch the football match last night?",
           "The laws of physics are complex."],
                batch_size=2))
