import pygame
import sys
import time
from pygame.locals import *
from random import randint
MOVE_SLEEP = 0.01
class MyTank:
    width = 600
    heights = 500
    speed = 10
    screen = 0
    myshells = []
    enemylist = []
    enemyshells = []
    grade = 0
    life = 3
    cnt = 0
    def startgame(self):
        pygame.init()
        self.screen = pygame.display.set_mode((self.width,self.heights),0,32)
        pygame.display.set_caption("bit tank")
        self.tank = Tank(self.screen,275,450)
        for i in range(6):
            self.enemylist.append(EnmeyTank(self.screen))
        while True:
            key = pygame.key.get_pressed()
            self.screen.fill((0,0,0))
            if key[K_LEFT]:
                self.tank.move('L')
            elif key[K_RIGHT]:
                self.tank.move('R')
            elif key[K_UP]:
                self.tank.move('U')
            elif key[K_DOWN]:
                self.tank.move('D')
            self.get_event()
            for shell in self.myshells:
                if shell.move() == True:
                    self.myshells.remove(shell)
                shell.display()
                a = shell.hitTank()
                #子弹碰撞
                if a == True:
                    if self.life >0:
                        self.myshells.remove(shell)
                        self.grade += 1
            #mytank碰撞
            if self.tank.live == True:
                if self.tank.hitTank():
                    self.life -= 1
                    if self.life <=0:
                        self.tank.live =False
                    else:self.tank = Tank(self.screen,275,450)
            #mytanke 碰撞子弹
            if self.tank.live == True:
                if self.tank.hitShell():
                    self.life -= 1
                    if self.life <=0:
                        self.tank.live = False
                    else:self.tank=Tank(self.screen,275,450)
            #敌方子弹击中我方坦克
            # 游戏结束
            if self.life <=0:
                self.gotGamePrint()
            for enemy in self.enemylist:
                enemy.move()
                print('move')
                enemy.display()
            # 添加敌方子弹
            self.cnt += 1
            if self.cnt % 100 ==0:
                for enemy in self.enemylist:
                    self.enemyshells.append(enemy.fire())
            #判断敌方子弹碰撞
            for enemyshell in self.enemyshells:
                f = enemyshell.move()
                enemyshell.display()
                if f:
                    self.enemyshells.remove(enemyshell)
            if len(self.enemylist)<6:
                self.enemylist.append(EnmeyTank(self.screen))
            self.screen.blit(self.getGrade(),(5,5))
            self.tank.display()
            pygame.display.update()
            time.sleep(0.02)
    def get_event(self):
        for event in pygame.event.get():
            if event.type == KEYDOWN:
                if event.key == K_SPACE:
                    self.myshells.append(self.tank.fire())
                if event.key == K_ESCAPE:
                    pass
    def getGrade(self):
        text = pygame.font.Font('./font/msyhbd.ttc',20).render("分数:{} 生命:{}".format(self.grade,self.life),True,(0,255,0))
        return text
    def gotGamePrint(self):
        text = pygame.font.Font('./font/msyh.ttc',70).render('game over!',True,(0,255,0))
        self.screen.blit(text,(100,200))
class Shell:
    width = 48
    height = 48
    live = True
    speed = 3
    def __init__(self,screen,tank):
        self.screen = screen
        self.image = pygame.image.load('./images/3.png')
        self.direction = tank.direction
        self.rect = self.image.get_rect()
        self.rect.left = tank.rect.left + (tank.width-self.width)/2.0+18
        # print(tank.rect.left,tank.width,self.width)
        self.rect.top = tank.rect.top + (tank.height - self.height)/2.0
        self.live = True
    def move(self):
        tag = self.isObstacle()
        if self.live == True:
            if self.direction == 'L' and self.direction not in tag:
                self.rect.left -= self.speed
            elif self.direction == 'R' and self.direction not in tag:
                self.rect.left += self.speed
            elif self.direction == 'U' and self.direction not in tag:
                self.rect.top -= self.speed
            elif self.direction == 'D' and self.direction not in tag:
                self.rect.top += self.speed
            else:
                pass
            if self.direction in tag:
                return True
            else:
                return False
        else:
            pass
    def display(self):
        # print(self.rect.left,self.rect.top)
        if self.live == True:
            self.screen.blit(self.image,self.rect)
    def isObstacle(self):
        tag = []
        if self.rect.left <=0:tag.append('L')
        if self.rect.left + self.width >= MyTank.width:tag.append('R')
        if self.rect.top <=0:tag.append('U')
        if self.rect.top + self.height  >=MyTank.heights:tag.append('D')
        return tag
    def hitTank(self):
        hitList = pygame.sprite.spritecollide(self,MyTank.enemylist,False)
        for e in hitList:
            e.live = False
            MyTank.enemylist.remove(e)
            self.live = False
            return True
        return False
    def hitMytank(self):
        hitList = pygame.sprite.spritecollide(self,MyTank.tank,False)
        for e in hitList:
            e.live = False
            MyTank.life -= 1
            return True
class BaseTank:
    width = 50
    height = 50
    direction = 'U'
    live = True
    time = 0
    images =  {}
    def __init__(self,screen,left,top):
        self.screen = screen
        self.images['L'] = pygame.image.load("images/04.jpg")
        self.images['R'] = pygame.image.load("images/02.jpg")
        self.images['U'] = pygame.image.load("images/01.jpg")
        self.images['D'] = pygame.image.load("images/03.jpg")
        self.image = self.images[self.direction]
        self.rect = self.image.get_rect()
        self.rect.left = left
        self.rect.top = top
        self.live = True  # 坦克是否被消灭
    def isObstacle(self):
        tag = []
        if self.rect.left <= 0: tag.append('L')
        if self.rect.left + self.width >= MyTank.width: tag.append('R')
        if self.rect.top <= 0: tag.append('U')
        if self.rect.top + self.height >= MyTank.heights: tag.append('D')
        return tag
    def display(self):
        if self.live == True:
            self.image = self.images[self.direction]
            self.screen.blit(self.image, self.rect)
    def fire(self):
        m = Shell(self.screen,self)
        return m
class Tank(BaseTank):
    images = {}
    def __init__(self,screen,left,top):
        super().__init__(screen,275,450)
        self.screen = screen
        self.speed = 2
        self.images['L'] = pygame.image.load('./images/4.jpg')
        self.images['R'] = pygame.image.load('./images/2.jpg')
        self.images['U'] = pygame.image.load('./images/1.jpg')
        self.images['D'] = pygame.image.load('./images/3.jpg')
        self.image = self.images[self.direction]
        self.rect = self.image.get_rect()
        self.rect.top = top
        self.rect.left = left
    def move(self, direction):
        if self.live == True:
            tag = self.isObstacle()
            if direction == self.direction:
                if self.direction == 'L' and self.direction not in tag:
                    self.rect.left -= self.speed
                elif self.direction == 'R' and self.direction not in tag:
                    self.rect.left += self.speed
                elif self.direction == 'U' and self.direction not in tag:
                    self.rect.top -= self.speed
                elif self.direction == 'D' and self.direction not in tag:
                    self.rect.top += self.speed
                else:
                    pass
            else:
                self.direction = direction
    def hitTank(self):
        hitList = pygame.sprite.spritecollide(self,MyTank.enemylist,False)
        for e in hitList:
            self.live = False
            return True
        return False
    def hitShell(self):
        hitlist = pygame.sprite.spritecollide(self, MyTank.enemyshells, False)
        for e in hitlist:
            self.live = False
            return True
        return False
class EnmeyTank(BaseTank):
    speed = 1
    def __init__(self,screen):
        super().__init__(screen,randint(1,5)*100,0)
        self.getdirection()
        self.step = 0
    def getdirection(self):
        self.direction = ['L','R','U','D'][randint(0,3)]
    def move(self):
        if self.live == True:
            if self.step == 0 or (self.direction in self.isObstacle()):
                self.getdirection()
                self.step = randint(0, 200)
            else:
                tag = self.isObstacle()
                if self.direction == 'L' and self.direction not in tag:
                    self.rect.left -= self.speed
                elif self.direction == 'R' and self.direction not in tag:
                    self.rect.left += self.speed
                elif self.direction == 'U' and self.direction not in tag:
                    self.rect.top -= self.speed
                elif self.direction == 'D' and self.direction not in tag:
                    self.rect.top += self.speed
                else:
                    pass
                self.step -= 1

if __name__ == '__main__':
    main = MyTank()
    main.startgame()
