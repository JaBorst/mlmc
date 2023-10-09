##########################################################################
#
# Example for  using a zero shot model (experimental feature)
#
##########################################################################
import mlmc

m = mlmc.models.zeroshot.Encoder(
    classes={"Sports":0, "Science":1},
)

m.predict(["Did you watch the football match last night?",
           "The laws of physics are complex."])
