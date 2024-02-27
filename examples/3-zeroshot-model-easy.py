##########################################################################
#
# Example for  using a zero shot model
#
##########################################################################
import mlmc

m = mlmc.models.zeroshot.Encoder(
    classes={"Sports":0, "Science":1},
)

print(m.predict(["Did you watch the football match last night?",
           "The laws of physics are complex."],
                batchsize=2))
